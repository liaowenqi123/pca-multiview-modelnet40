"""
V7 自包含训练脚本。
包含所有预处理、数据集、模型定义、训练逻辑，无需依赖 common/。

用法:
    python full_train.py --epochs 100 --batch_size 64

结果: ModelNet40 测试集 84%
"""

import os
import sys
import time
import math
import argparse
import warnings
from pathlib import Path

# ─── ROCm 环境 ───
os.environ.setdefault("MIOPEN_FIND_MODE", "normal")
os.environ.setdefault("MIOPEN_DEBUG_FIND_ONLY_SOLVER", "1")
os.environ.setdefault("MIOPEN_DEBUG_GCN_ASM_KERNELS", "0")
os.environ.setdefault("MIOPEN_DEBUG_CONV_DIRECT", "0")
os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")
os.environ.setdefault("MIOPEN_DEBUG_CONV_FFT", "0")
os.environ.setdefault("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM", "0")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    torch.backends.cudnn.enabled = False
except Exception:
    pass

# ─── torchvision 镜像 ───
import torch.hub as hub
_orig_download = hub.download_url_to_file


def _mirror_download(url, dst, hash_prefix=None, progress=True):
    try:
        return _orig_download(url, dst, hash_prefix=hash_prefix, progress=progress)
    except Exception:
        alt = url.replace("https://download.pytorch.org/models/",
                          "https://hf-mirror.com/pytorch/vision/resolve/main/")
        if alt == url:
            raise
        return _orig_download(alt, dst, hash_prefix=hash_prefix, progress=progress)


hub.download_url_to_file = _mirror_download

# ══════════════════════════════════════════════════════════════
#  常量
# ══════════════════════════════════════════════════════════════

NUM_CLASSES = 40
NUM_POINTS = 2048
GRID_SIZE = 224
DATA_DIR = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ══════════════════════════════════════════════════════════════
#  PCA 对齐
# ══════════════════════════════════════════════════════════════

def pca_align(points):
    N = points.shape[0]
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid
    cov = (centered.T @ centered) / N
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    aligned = centered @ eigenvectors
    for i in range(3):
        if np.median(aligned[:, i]) < 0:
            aligned[:, i] *= -1
    max_abs = np.max(np.abs(aligned))
    if max_abs > 1e-8:
        aligned = aligned / max_abs
    return aligned


# ══════════════════════════════════════════════════════════════
#  三视图投影
# ══════════════════════════════════════════════════════════════

def _scatter_project(points_2d, values_density, values_max, values_min, grid_size=GRID_SIZE):
    coords = ((points_2d + 1.0) / 2.0 * (grid_size - 1)).astype(np.int32)
    coords = np.clip(coords, 0, grid_size - 1)
    x_idx, y_idx = coords[:, 0], coords[:, 1]
    flat_idx = y_idx * grid_size + x_idx

    density = np.zeros(grid_size * grid_size, dtype=np.float32)
    np.add.at(density, flat_idx, values_density.astype(np.float32))

    max_depth = np.full(grid_size * grid_size, -np.inf, dtype=np.float32)
    np.maximum.at(max_depth, flat_idx, values_max.astype(np.float32))

    min_depth = np.full(grid_size * grid_size, np.inf, dtype=np.float32)
    np.minimum.at(min_depth, flat_idx, values_min.astype(np.float32))

    density = density.reshape(grid_size, grid_size)
    max_depth = max_depth.reshape(grid_size, grid_size)
    min_depth = min_depth.reshape(grid_size, grid_size)

    max_depth[np.isneginf(max_depth)] = 0.0
    min_depth[np.isposinf(min_depth)] = 0.0

    d_max = density.max()
    if d_max > 0:
        density /= d_max

    max_depth = np.clip((max_depth + 1.0) / 2.0, 0.0, 1.0)
    min_depth = np.clip((min_depth + 1.0) / 2.0, 0.0, 1.0)

    return np.stack([density, max_depth, min_depth], axis=0).astype(np.float32)


def multi_view_project_6(points, grid_size=GRID_SIZE):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ones = np.ones(points.shape[0], dtype=np.float32)

    xy_pos = _scatter_project(np.stack([x, y], axis=1), ones, z, z, grid_size)
    xy_neg = _scatter_project(np.stack([x, y], axis=1), ones, -z, -z, grid_size)
    xz_pos = _scatter_project(np.stack([x, z], axis=1), ones, y, y, grid_size)
    xz_neg = _scatter_project(np.stack([x, z], axis=1), ones, -y, -y, grid_size)
    yz_pos = _scatter_project(np.stack([y, z], axis=1), ones, x, x, grid_size)
    yz_neg = _scatter_project(np.stack([y, z], axis=1), ones, -x, -x, grid_size)

    return np.stack([xy_pos, xy_neg, xz_pos, xz_neg, yz_pos, yz_neg], axis=0)


# ══════════════════════════════════════════════════════════════
#  数据增强
# ══════════════════════════════════════════════════════════════

def augment_points(points):
    theta = np.random.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_z = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
    points = points @ rot_z.T
    points *= np.random.uniform(0.9, 1.1)
    points += np.random.uniform(-0.05, 0.05, size=(1, 3))
    points += np.random.normal(0, 0.01, size=points.shape).astype(np.float32)
    max_abs = np.max(np.abs(points))
    if max_abs > 0:
        points /= max_abs
    return points


# ══════════════════════════════════════════════════════════════
#  数据集
# ══════════════════════════════════════════════════════════════

class ModelNet40MultiView(Dataset):
    def __init__(self, points, labels, augment=False, cache_pca=True, return_aligned=True):
        self.points = points
        self.labels = labels
        self.augment = augment
        self.cache_pca = cache_pca
        self.return_aligned = return_aligned

        if cache_pca:
            print(f"[dataset] PCA 对齐 ({len(points)} 样本)...")
            self.aligned = []
            for i in range(len(points)):
                self.aligned.append(pca_align(points[i]))
                if (i + 1) % 1000 == 0:
                    print(f"  PCA: {i+1}/{len(points)}")
            print("[dataset] PCA 完成")
        else:
            self.aligned = None

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        if self.aligned is not None:
            pts = self.aligned[idx].copy()
        else:
            pts = pca_align(self.points[idx])

        if self.augment:
            pts = augment_points(pts)

        views = multi_view_project_6(pts, GRID_SIZE)
        views = (views - IMAGENET_MEAN) / IMAGENET_STD

        label = self.labels[idx]
        if self.return_aligned:
            return (torch.from_numpy(views), torch.from_numpy(pts),
                    torch.tensor(label, dtype=torch.long))
        return torch.from_numpy(views), torch.tensor(label, dtype=torch.long)


# ══════════════════════════════════════════════════════════════
#  PointNetPathV7
# ══════════════════════════════════════════════════════════════

class PointNetPathV7(nn.Module):
    def __init__(self, in_dim=3, d_latent=64, d_pointnet=512, d_kv=128,
                 n_sample=512, c_out_v3=512, num_classes=NUM_CLASSES):
        super().__init__()
        self.d_latent = d_latent
        self.d_kv = d_kv
        self.n_sample = n_sample

        self.compress = nn.Sequential(
            nn.Conv1d(in_dim, 32, 1), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, d_latent, 1), nn.BatchNorm1d(d_latent), nn.ReLU(inplace=True),
        )
        self.sa_q = nn.Linear(d_latent, d_latent)
        self.sa_k = nn.Linear(d_latent, d_latent)
        self.sa_v = nn.Linear(d_latent, d_latent)
        self.sa_norm = nn.LayerNorm(d_latent)

        self.expand = nn.Sequential(
            nn.Conv1d(d_latent, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, d_pointnet, 1), nn.BatchNorm1d(d_pointnet), nn.ReLU(inplace=True),
        )
        self.kv_proj = nn.Conv1d(d_pointnet, d_kv * 2, 1)
        self.q_proj = nn.Linear(c_out_v3, d_kv)
        self.out_proj = nn.Sequential(
            nn.Linear(d_kv, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, points, v3_view_features):
        B, N, _ = points.shape
        n = min(N, self.n_sample)
        if self.training or n < N:
            idx = torch.randperm(N, device=points.device)[:n]
            idx = idx.unsqueeze(0).expand(B, -1)
            points = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

        x = points.transpose(1, 2)
        x = self.compress(x)
        x = x.transpose(1, 2)

        q = self.sa_q(x); k = self.sa_k(x); v = self.sa_v(x)
        attn = (q @ k.transpose(-2, -1)) / (self.d_latent ** 0.5)
        attn = F.softmax(attn, dim=-1)
        x = self.sa_norm(attn @ v + x)

        x = x.transpose(1, 2)
        x = self.expand(x)

        kv = self.kv_proj(x)
        k_pn = kv[:, :self.d_kv, :]
        v_pn = kv[:, self.d_kv:, :]
        q_v3 = self.q_proj(v3_view_features)

        scores = torch.bmm(q_v3, k_pn) / (self.d_kv ** 0.5)
        attn_w = F.softmax(scores, dim=-1)
        weighted_v = torch.bmm(attn_w, v_pn.transpose(1, 2))
        pooled = weighted_v.mean(dim=1)

        return self.out_proj(pooled)


# ══════════════════════════════════════════════════════════════
#  FusionGate
# ══════════════════════════════════════════════════════════════

class FusionGate(nn.Module):
    def __init__(self, pure_mul=False):
        super().__init__()
        self.pure_mul = pure_mul
        if not pure_mul:
            self.w_raw = nn.Parameter(torch.tensor(0.0))
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, f1, f2):
        mult = torch.sigmoid(f1) * torch.sigmoid(f2)
        if self.pure_mul:
            score = mult
        else:
            w = torch.sigmoid(self.w_raw)
            score = w * mult + (1 - w) * (f1 + f2)
        return score * torch.exp(self.log_temp)


# ══════════════════════════════════════════════════════════════
#  MultiViewResNetV7
# ══════════════════════════════════════════════════════════════

def _make_backbone(name="resnet18", pretrained=True):
    import torchvision.models as models
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        bb = resnet18(weights=w)
    except Exception:
        bb = models.resnet18(pretrained=pretrained)
    return nn.Sequential(
        bb.conv1, bb.bn1, bb.relu, bb.maxpool,
        bb.layer1, bb.layer2, bb.layer3, bb.layer4,
    )


class MultiViewResNetV7(nn.Module):
    def __init__(self, backbone="resnet18", num_views=6, symmetric_fusion=True,
                 dim_reduce=256, freeze_until="layer2",
                 d_latent=64, d_pointnet=512, d_kv=128, n_sample=512,
                 pure_mul=False, confidence_bias=-1.0, pretrained=True,
                 dropout=0.5, num_classes=NUM_CLASSES):
        super().__init__()
        C_out = 512
        self.backbone = _make_backbone(backbone, pretrained=pretrained)
        self.backbone_out_channels = C_out
        self.num_views = num_views
        self.symmetric_fusion = symmetric_fusion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_fused = 3 if symmetric_fusion else num_views

        # 冻结
        _idx = {"conv1": 0, "bn1": 1, "layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7}
        _order = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        ids = set()
        for nm in _order:
            ids.add(_idx[nm])
            if nm == freeze_until:
                break
        frozen_count = 0
        for nm, p in self.backbone.named_parameters():
            try:
                if int(nm.split(".")[0]) in ids:
                    p.requires_grad = False
                    frozen_count += 1
            except ValueError:
                pass
        print(f"[V7] 冻结 {frozen_count} backbone 参数 (<= {freeze_until})")

        # V3 组件
        self.view_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(C_out, dim_reduce),
                          nn.BatchNorm1d(dim_reduce), nn.ReLU(inplace=True))
            for _ in range(self.num_fused)
        ])
        fusion_in = self.num_fused * dim_reduce
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.confidence_proj = nn.Linear(C_out, 1)
        nn.init.constant_(self.confidence_proj.bias, confidence_bias)
        self.view_classifiers = nn.ModuleList([
            nn.Linear(C_out, num_classes) for _ in range(self.num_fused)
        ])

        # PointNet 路径
        self.point_path = PointNetPathV7(
            in_dim=3, d_latent=d_latent, d_pointnet=d_pointnet, d_kv=d_kv,
            n_sample=n_sample, c_out_v3=C_out, num_classes=num_classes,
        )

        # 融合门
        self.fusion_gate = FusionGate(pure_mul=pure_mul)

    def forward(self, x):
        views, points = x
        B, V, C_img, H, W = views.shape
        xf = views.view(B * V, C_img, H, W)
        f = self.backbone(xf)
        f = self.avgpool(f).view(B * V, -1)
        f = f.view(B, V, -1)

        if self.symmetric_fusion:
            f = torch.maximum(f[:, 0::2, :], f[:, 1::2, :])

        v3_vf = f
        proj = [self.view_projections[i](f[:, i, :]) for i in range(self.num_fused)]
        logits_fus = self.classifier(torch.cat(proj, dim=1))
        logits_v = []
        for i in range(self.num_fused):
            fi = f[:, i, :]
            conf = torch.sigmoid(self.confidence_proj(fi))
            logits_v.append(conf * self.view_classifiers[i](fi))
        f1 = logits_fus + torch.stack(logits_v, dim=1).sum(dim=1)
        f2 = self.point_path(points, v3_vf)
        return self.fusion_gate(f1, f2)


# ══════════════════════════════════════════════════════════════
#  训练工具
# ══════════════════════════════════════════════════════════════

def _unpack_batch(batch, device):
    if len(batch) == 3:
        views, points, labels = batch
        return ((views.to(device), points.to(device)),
                labels.to(device))
    views, labels = batch
    return (views.to(device), labels.to(device))


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    try:
        from tqdm import tqdm
        pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
    except ImportError:
        pbar = loader

    for batch in pbar:
        inp, labels = _unpack_batch(batch, device)
        optimizer.zero_grad()
        logits = model(inp)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        inp, labels = _unpack_batch(batch, device)
        logits = model(inp)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V7 standalone train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dim_reduce", type=int, default=256)
    parser.add_argument("--freeze_until", type=str, default="layer2")
    parser.add_argument("--d_latent", type=int, default=64)
    parser.add_argument("--d_pointnet", type=int, default=512)
    parser.add_argument("--d_kv", type=int, default=128)
    parser.add_argument("--n_sample", type=int, default=512)
    parser.add_argument("--pure_mul", action="store_true")
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--no_cache_pca", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[设备] {device}")

    train_points = np.load(DATA_DIR / "train_points.npy")
    train_labels = np.load(DATA_DIR / "train_labels.npy")
    test_points = np.load(DATA_DIR / "test_points.npy")

    test_labels_path = DATA_DIR / "test_labels.npy"
    if test_labels_path.exists():
        test_labels = np.load(test_labels_path)
    else:
        np.random.seed(42)
        indices = np.random.permutation(len(train_points))
        split = int(len(train_points) * 0.8)
        train_idx, val_idx = indices[:split], indices[split:]
        test_points, test_labels = train_points[val_idx], train_labels[val_idx]
        train_points, train_labels = train_points[train_idx], train_labels[train_idx]
        print(f"[数据] 训练 {len(train_points)} / 验证 {len(test_points)}")

    print(f"[数据] 训练: {len(train_points)}, 验证: {len(test_points)}")

    train_dataset = ModelNet40MultiView(
        train_points, train_labels, augment=True,
        cache_pca=not args.no_cache_pca, return_aligned=True)
    test_dataset = ModelNet40MultiView(
        test_points, test_labels, augment=False,
        cache_pca=not args.no_cache_pca, return_aligned=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"))

    model = MultiViewResNetV7(
        backbone="resnet18", num_views=6, symmetric_fusion=True,
        dim_reduce=args.dim_reduce, freeze_until=args.freeze_until,
        d_latent=args.d_latent, d_pointnet=args.d_pointnet, d_kv=args.d_kv,
        n_sample=args.n_sample, pure_mul=args.pure_mul,
        pretrained=not args.no_pretrain, dropout=args.dropout,
    ).to(device)

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 冻结: {frozen:,} | 可训练: {trainable:,}")

    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.view_projections.parameters(), "lr": args.lr * 0.1},
        {"params": model.classifier.parameters(), "lr": args.lr},
        {"params": model.confidence_proj.parameters(), "lr": args.lr},
        {"params": model.view_classifiers.parameters(), "lr": args.lr},
        {"params": model.point_path.parameters(), "lr": args.lr},
        {"params": model.fusion_gate.parameters(), "lr": args.lr * 0.01},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_epoch = 0
    best_acc = 0.0
    ckpt_dir = DATA_DIR / "checkpoints" / "v7"
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_acc = ckpt.get("best_acc", 0.0)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"V7 | {args.epochs} eps | bs={args.batch_size} | lr={args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_acc": best_acc,
            }, ckpt_dir / "best_model.pth")
            print(f"  >>> 保存最佳模型 (acc={best_acc:.4f})")

        if (epoch + 1) % 20 == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "best_acc": best_acc,
            }, ckpt_dir / f"epoch_{epoch+1}.pth")

    print(f"\n{'='*60}")
    print(f"训练结束。最佳: {best_acc:.4f}")
    if not args.pure_mul:
        w = torch.sigmoid(model.fusion_gate.w_raw).item()
        T = torch.exp(model.fusion_gate.log_temp).item()
        print(f"FusionGate: w={w:.3f}  T={T:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
