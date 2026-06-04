"""
ModelNet40 点云分类 — PyTorch 实现
=====================================
方案：
  1. PCA 对齐：方差最大 → x 轴，次大 → y 轴，最小 → z 轴
  2. 三视图投影：xy / xz / yz 各生成 224×224×3 密度-深度图
  3. 共享 ResNet50（ImageNet 预训练）提取三视图特征
  4. 拼接三路特征 → FC → 40 类输出

用法：
  python train.py --epochs 100 --batch_size 32 --lr 0.001

依赖（不修改已有环境，仅需额外安装）：
  pip install tqdm tensorboard  （可选，用于进度条和日志）
"""

import os

# ─── 修复 Windows + ROCm 下 MIOpen HIPRTC 编译 ─────────────────
# 问题：MIOpen JIT 编译 BatchNorm kernel 时找不到 <type_traits> 等 C++ 头文件
# 解决：强制 PyTorch 使用 ATen 原生实现，不走 MIOpen 即时编译路径
os.environ.setdefault("MIOPEN_FIND_MODE", "normal")             # 减少 JIT 编译
os.environ.setdefault("MIOPEN_DEBUG_FIND_ONLY_SOLVER", "1")     # 只用预编译 solver
os.environ.setdefault("MIOPEN_DEBUG_GCN_ASM_KERNELS", "0")      # 禁用汇编 kernel
os.environ.setdefault("MIOPEN_DEBUG_CONV_DIRECT", "0")
os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")
os.environ.setdefault("MIOPEN_DEBUG_CONV_FFT", "0")
os.environ.setdefault("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM", "0")
# 如果 MIOpen 仍然失败，让 PyTorch 回退到自身实现
os.environ.setdefault("MIOPEN_ENABLE_LOGGING_CMD", "0")

import sys
import time
import math
import warnings
import argparse
from pathlib import Path

import numpy as np
import torch

# ─── 最终防线：如果环境变量不够，直接禁用 MIOpen/cuDNN 后端 ───
# 在 ROCm 上 torch.backends.cudnn 映射到 MIOpen；禁用后走 ATen 原生 HIP kernel
try:
    torch.backends.cudnn.enabled = False
    print("[rocm] 已禁用 cuDNN/MIOpen 后端，使用 ATen 原生实现。")
except Exception:
    pass

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# ─── 镜像下载 ResNet50 权重 ───────────────────────────────────────────
# 国内用户通过 HF 镜像加速下载 torchvision 预训练模型

def _setup_mirror_for_torchvision():
    """Monkey-patch torch.hub 的下载函数，优先走 HF 镜像。"""
    try:
        import torch.hub as hub
        _orig_download = hub.download_url_to_file

        def _mirror_download(url, dst, hash_prefix=None, progress=True):
            # 尝试原始 URL，失败后走 HF 镜像
            import urllib.error
            try:
                return _orig_download(url, dst, hash_prefix=hash_prefix, progress=progress)
            except (urllib.error.URLError, OSError) as e:
                # download.pytorch.org → hf-mirror.com 的 pytorch/vision 镜像
                alt_url = url.replace(
                    "https://download.pytorch.org/models/",
                    "https://hf-mirror.com/pytorch/vision/resolve/main/",
                )
                if alt_url != url:
                    print(f"[mirror] 主站下载失败，尝试镜像: {alt_url}")
                else:
                    raise e
                try:
                    return _orig_download(alt_url, dst, hash_prefix=hash_prefix, progress=progress)
                except Exception:
                    # 最后尝试直接走原站（可能是其他原因失败）
                    raise e

        hub.download_url_to_file = _mirror_download

        # 同时 patch load_state_dict_from_url（它内部调用 download_url_to_file）
        if hasattr(hub, "load_state_dict_from_url"):
            _orig_load = hub.load_state_dict_from_url

            def _mirror_load(url, *args, **kwargs):
                alt_url = url.replace(
                    "https://download.pytorch.org/models/",
                    "https://hf-mirror.com/pytorch/vision/resolve/main/",
                )
                # 先走原始链接（已通过上面的 patch 做镜像 fallback）
                return _orig_load(url, *args, **kwargs)

            hub.load_state_dict_from_url = _mirror_load

        print("[mirror] torchvision 下载已配置 HF 镜像回退。")
    except Exception:
        pass  # 非关键，失败了就正常走原站


_setup_mirror_for_torchvision()

# ─── 常量 ────────────────────────────────────────────────────────────
NUM_CLASSES = 40
NUM_POINTS = 2048
GRID_SIZE = 224          # ResNet50 标准输入
CACHE_DIR = Path(__file__).parent / ".cache"


# ═══════════════════════════════════════════════════════════════════════
#  第 1 部分：PCA 对齐
# ═══════════════════════════════════════════════════════════════════════

def pca_align(points: np.ndarray) -> np.ndarray:
    """
    对点云做 PCA 对齐。
    
    步骤：
      1. 去中心化（减质心）
      2. 计算协方差矩阵 (3×3)
      3. 特征分解，按特征值降序排列
         - 最大特征值对应的方向 → x 轴
         - 次大特征值对应的方向 → y 轴
         - 最小特征值对应的方向 → z 轴
      4. 旋转点云（乘以特征向量矩阵的转置）
      5. 处理方向模糊性：确保每轴分布偏正半轴
      6. 归一化到 [-1, 1]³

    Args:
        points: (N, 3) 原始点云
    
    Returns:
        aligned: (N, 3) 对齐且归一化后的点云
    """
    N = points.shape[0]

    # 1. 去中心化
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid  # (N, 3)

    # 2. 协方差矩阵
    cov = (centered.T @ centered) / N  # (3, 3)

    # 3. 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # eigh 按特征值升序返回

    # 降序排列：最大 → x, 次大 → y, 最小 → z
    order = np.argsort(eigenvalues)[::-1]            # [2, 1, 0] → 最大的在前
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]             # (3, 3)，每列是一个特征向量

    # 保证右手坐标系（行列式 > 0）
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    # 4. 旋转
    aligned = centered @ eigenvectors  # (N, 3)

    # 5. 方向模糊性处理：确保每轴的大多数点位于正半轴
    for i in range(3):
        if np.median(aligned[:, i]) < 0:
            aligned[:, i] *= -1

    # 6. 归一化到 [-1, 1]³
    # 使用 max(abs) 做各轴统一缩放，保持比例
    max_abs = np.max(np.abs(aligned))
    if max_abs > 1e-8:
        aligned = aligned / max_abs

    return aligned


# ═══════════════════════════════════════════════════════════════════════
#  第 2 部分：三视图投影
# ═══════════════════════════════════════════════════════════════════════

def _scatter_project(points_2d: np.ndarray,  # (M, 2)  坐标在 [-1, 1]
                     values_density: np.ndarray,  # (M,)   密度值（全1）
                     values_max: np.ndarray,      # (M,)   深度值
                     values_min: np.ndarray,      # (M,)   深度值
                     grid_size: int = GRID_SIZE,
                     ) -> np.ndarray:
    """
    将 2D 点散射到 grid_size×grid_size 的网格上，生成三通道图像。
    
    通道 0：密度（归一化点计数）
    通道 1：最大深度值
    通道 2：最小深度值

    Args:
        points_2d: (M, 2) 投影平面坐标，范围 [-1, 1]
        values_density: (M,) 密度权重（通常全1）
        values_max: (M,) 深度值（用于 max 通道）
        values_min: (M,) 深度值（用于 min 通道）
        grid_size: 网格分辨率
    
    Returns:
        image: (3, grid_size, grid_size) float32，值域 [0, 1]
    """
    # 坐标映射到 [0, grid_size-1] 的整数索引
    coords = ((points_2d + 1.0) / 2.0 * (grid_size - 1)).astype(np.int32)
    coords = np.clip(coords, 0, grid_size - 1)

    x_idx = coords[:, 0]
    y_idx = coords[:, 1]
    flat_idx = y_idx * grid_size + x_idx  # (M,)

    # 通道 0：密度 — 使用 scatter_add
    density = np.zeros(grid_size * grid_size, dtype=np.float32)
    np.add.at(density, flat_idx, values_density.astype(np.float32))

    # 通道 1：最大深度 — 使用 scatter max
    max_depth = np.full(grid_size * grid_size, -np.inf, dtype=np.float32)
    np.maximum.at(max_depth, flat_idx, values_max.astype(np.float32))

    # 通道 2：最小深度 — 使用 scatter min
    min_depth = np.full(grid_size * grid_size, np.inf, dtype=np.float32)
    np.minimum.at(min_depth, flat_idx, values_min.astype(np.float32))

    # reshape
    density = density.reshape(grid_size, grid_size)
    max_depth = max_depth.reshape(grid_size, grid_size)
    min_depth = min_depth.reshape(grid_size, grid_size)

    # 处理空 cell
    max_depth[np.isneginf(max_depth)] = 0.0
    min_depth[np.isposinf(min_depth)] = 0.0

    # 归一化密度：除以最大值（避免除零）
    d_max = density.max()
    if d_max > 0:
        density = density / d_max

    # 归一化深度：假设深度值在 [-1, 1] 范围内，映射到 [0, 1]
    # 对于空 cell（值为0），保持0
    max_depth = np.clip((max_depth + 1.0) / 2.0, 0.0, 1.0)
    min_depth = np.clip((min_depth + 1.0) / 2.0, 0.0, 1.0)

    # 堆叠为 (3, H, W)
    image = np.stack([density, max_depth, min_depth], axis=0).astype(np.float32)
    return image


def multi_view_project(points: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    生成三视图投影图像。

    Args:
        points: (N, 3) 已对齐归一化的点云，坐标范围 [-1, 1]
        grid_size: 图像分辨率
    
    Returns:
        views: (3, 3, grid_size, grid_size)
               维度 0 → 三个视图 [xy, xz, yz]
               维度 1 → 通道 [密度, max深度, min深度]
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # 密度权重（全1，即每个点贡献相同）
    ones = np.ones(points.shape[0], dtype=np.float32)

    # 视图 1：xy 投影（俯视图），z 为深度
    xy_view = _scatter_project(
        points_2d=np.stack([x, y], axis=1),
        values_density=ones,
        values_max=z,
        values_min=z,
        grid_size=grid_size,
    )

    # 视图 2：xz 投影（前视图），y 为深度
    xz_view = _scatter_project(
        points_2d=np.stack([x, z], axis=1),
        values_density=ones,
        values_max=y,
        values_min=y,
        grid_size=grid_size,
    )

    # 视图 3：yz 投影（侧视图），x 为深度
    yz_view = _scatter_project(
        points_2d=np.stack([y, z], axis=1),
        values_density=ones,
        values_max=x,
        values_min=x,
        grid_size=grid_size,
    )

    views = np.stack([xy_view, xz_view, yz_view], axis=0)  # (3, 3, H, W)
    return views


# ═══════════════════════════════════════════════════════════════════════
#  第 3 部分：数据集
# ═══════════════════════════════════════════════════════════════════════

class ModelNet40MultiView(Dataset):
    """ModelNet40 三视图数据集，带 PCA 对齐 + 实时投影。"""

    def __init__(self, points: np.ndarray, labels: np.ndarray,
                 augment: bool = False, cache_pca: bool = True):
        """
        Args:
            points: (N, 2048, 3) 点云
            labels: (N,) 标签
            augment: 是否做数据增强
            cache_pca: 是否缓存 PCA 对齐结果（推荐 True）
        """
        self.points = points
        self.labels = labels
        self.augment = augment
        self.cache_pca = cache_pca

        # 缓存 PCA 对齐后的点云（减少重复计算）
        if cache_pca:
            print(f"[dataset] 正在进行 PCA 对齐（{len(points)} 个样本）...")
            self.aligned = []
            for i in range(len(points)):
                self.aligned.append(pca_align(points[i]))
                if (i + 1) % 1000 == 0:
                    print(f"  PCA 对齐进度: {i+1}/{len(points)}")
            print("[dataset] PCA 对齐完成。")
        else:
            self.aligned = None
            print("[dataset] PCA 对齐将在 __getitem__ 中实时计算。")

        # ImageNet 归一化参数（与预训练 ResNet 一致）
        self.normalize_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.normalize_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        # 获取对齐后的点云
        if self.aligned is not None:
            pts = self.aligned[idx].copy()
        else:
            pts = pca_align(self.points[idx])

        # 数据增强（仅在训练时）
        if self.augment:
            pts = self._augment(pts)

        # 三视图投影
        views = multi_view_project(pts, GRID_SIZE)  # (3, 3, 224, 224)

        # ImageNet 归一化
        views = (views - self.normalize_mean) / self.normalize_std

        label = self.labels[idx]
        return torch.from_numpy(views), torch.tensor(label, dtype=torch.long)

    def _augment(self, points: np.ndarray) -> np.ndarray:
        """点云数据增强：随机旋转（绕 z 轴）、缩放、平移、抖动。"""
        # 绕 z 轴随机旋转（PCA 对齐后 z 轴方差最小，绕它旋转合理）
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_z = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1],
        ], dtype=np.float32)
        points = points @ rot_z.T

        # 随机缩放
        scale = np.random.uniform(0.9, 1.1)
        points = points * scale

        # 随机平移
        shift = np.random.uniform(-0.05, 0.05, size=(1, 3))
        points = points + shift

        # 坐标抖动
        jitter = np.random.normal(0, 0.01, size=points.shape).astype(np.float32)
        points = points + jitter

        # 重新归一化到 [-1, 1]
        max_abs = np.max(np.abs(points))
        if max_abs > 0:
            points = points / max_abs

        return points


# ═══════════════════════════════════════════════════════════════════════
#  第 4 部分：模型
# ═══════════════════════════════════════════════════════════════════════

class MultiViewResNet(nn.Module):
    """
    三视图 ResNet50 特征提取 + 融合分类。

    结构：
      Input: 3 张 3×224×224 图像
        → 共享 ResNet50（去除最后的 FC + avgpool）
        → 每张图得到 (2048, 7, 7) 特征图
        → AdaptiveAvgPool2d(1) → 各 (2048,) 向量
        → 拼接 → (6144,)
        → FC → (512,) → ReLU → Dropout
        → FC → (40,) 分类输出
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 dropout: float = 0.5):
        super().__init__()

        # 加载 ResNet50 骨干网络
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V1
            else:
                weights = None
            backbone = resnet50(weights=weights)
        except (ImportError, AttributeError):
            # 兼容旧版 torchvision
            import torchvision.models as models
            backbone = models.resnet50(pretrained=pretrained)

        # 去掉最后的 FC 层和 avgpool（保留到 layer4 输出）
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.backbone_out_channels = 2048

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类头
        fused_dim = self.backbone_out_channels * 3  # 6144
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 3, 224, 224)
                维度 1 → 三个视图（0:xy, 1:xz, 2:yz）
                维度 2 → 通道 RGB
        
        Returns:
            logits: (B, num_classes)
        """
        B = x.size(0)
        # 将三个视图合并到 batch 维度，共享 backbone
        # (B, 3, 3, 224, 224) → (B*3, 3, 224, 224)
        x_flat = x.view(B * 3, 3, GRID_SIZE, GRID_SIZE)

        # 共享特征提取
        features = self.backbone(x_flat)          # (B*3, 2048, 7, 7)
        features = self.avgpool(features)          # (B*3, 2048, 1, 1)
        features = features.view(B * 3, -1)        # (B*3, 2048)

        # 恢复三视图维度并拼接
        features = features.view(B, 3, -1)         # (B, 3, 2048)
        fused = features.view(B, -1)                # (B, 6144)

        # 分类
        logits = self.classifier(fused)             # (B, num_classes)
        return logits


# ═══════════════════════════════════════════════════════════════════════
#  第 5 部分：训练 / 评估 / 主函数
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, args):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # 可选进度条
    try:
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d} [train]", leave=False)
    except ImportError:
        pbar = dataloader

    for batch_idx, (views, labels) in enumerate(pbar):
        views = views.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(views)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "acc": f"{correct/total:.3f}"})

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for views, labels in dataloader:
        views = views.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(views)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    parser = argparse.ArgumentParser(description="ModelNet40 三视图分类 (PyTorch)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout 比率")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--no_pretrain", action="store_true", help="不使用预训练权重")
    parser.add_argument("--no_cache_pca", action="store_true", help="不缓存 PCA 结果")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda / cpu)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="模型保存目录")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    args = parser.parse_args()

    # ── 设备 ──
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] CUDA 不可用，回退到 CPU。")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[设备] 使用 {device}")

    # ── 加载数据 ──
    data_dir = Path(__file__).parent
    train_points = np.load(data_dir / "train_points.npy")
    train_labels = np.load(data_dir / "train_labels.npy")
    test_points = np.load(data_dir / "test_points.npy")

    # 检查是否有 test_labels
    test_labels_path = data_dir / "test_labels.npy"
    if test_labels_path.exists():
        test_labels = np.load(test_labels_path)
        print(f"[数据] 已加载 test_labels.npy ({len(test_labels)} 个标签)")
    else:
        # 从 train 中分出 20% 作为验证集
        print("[数据] 未找到 test_labels.npy，从训练集分出 20% 作为验证集。")
        np.random.seed(42)
        indices = np.random.permutation(len(train_points))
        split = int(len(train_points) * 0.8)
        train_idx, val_idx = indices[:split], indices[split:]
        test_points = train_points[val_idx]
        test_labels = train_labels[val_idx]
        train_points = train_points[train_idx]
        train_labels = train_labels[train_idx]
        print(f"[数据] 训练集 {len(train_points)} / 验证集 {len(test_points)}")

    print(f"[数据] 训练样本: {len(train_points)}, 测试样本: {len(test_points)}")
    print(f"[数据] 类别数: {len(np.unique(train_labels))}")

    # ── 数据集 ──
    train_dataset = ModelNet40MultiView(
        train_points, train_labels,
        augment=True,
        cache_pca=not args.no_cache_pca,
    )
    test_dataset = ModelNet40MultiView(
        test_points, test_labels,
        augment=False,
        cache_pca=not args.no_cache_pca,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    # ── 模型 ──
    model = MultiViewResNet(
        num_classes=NUM_CLASSES,
        pretrained=not args.no_pretrain,
        dropout=args.dropout,
    )
    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数量: {total_params:,}  |  可训练: {trainable_params:,}")

    # ── 优化器 & 调度器 ──
    # 对 backbone 使用较小的学习率
    backbone_params = model.backbone.parameters()
    classifier_params = model.classifier.parameters()
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": classifier_params, "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── 恢复训练 ──
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"[恢复] 从 epoch {start_epoch} 恢复，最佳准确率 {best_acc:.4f}")

    # ── 创建 checkpoint 目录 ──
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── 训练循环 ──
    print(f"\n{'='*60}")
    print(f"开始训练 — {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1, args
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, os.path.join(args.checkpoint_dir, "best_model.pth"))
            print(f"  >>> 保存最佳模型 (acc={best_acc:.4f})")

        # 定期保存
        if (epoch + 1) % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, os.path.join(args.checkpoint_dir, f"epoch_{epoch+1}.pth"))

    print(f"\n{'='*60}")
    print(f"训练结束。最佳验证准确率: {best_acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
