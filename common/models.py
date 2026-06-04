"""
MultiViewResNet：多视角 ResNet 特征提取 + 对称融合 + 分类。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocessing import NUM_CLASSES, GRID_SIZE

# 可用的骨干网络
_BACKBONES = {
    "resnet18": (512,  "resnet18"),
    "resnet50": (2048, "resnet50"),
}


def _make_backbone(name: str, pretrained: bool = True):
    """构造指定版本的 ResNet 骨干（去除了最后的 FC + avgpool）。"""
    import torchvision.models as models
    try:
        from torchvision.models import (
            resnet18, resnet50,
            ResNet18_Weights, ResNet50_Weights,
        )
        wmap = {
            "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1 if pretrained else None),
            "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1 if pretrained else None),
        }
        fn, weights = wmap[name]
        backbone = fn(weights=weights)
    except (ImportError, AttributeError):
        backbone = getattr(models, name)(pretrained=pretrained)

    return nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )


class MultiViewResNet(nn.Module):
    """
    多视图 ResNet 分类器。

    参数：
        backbone:      "resnet18" 或 "resnet50"
        num_views:     3 或 6（6 时必须启用 symmetric_fusion）
        symmetric_fusion: 是否对成对的对称视图做 max-pool 融合
        pretrained:    是否使用 ImageNet 预训练权重
        dropout:       Dropout 比率 (default 0.5)

    输入形状：
        3 视图:  (B, 3, 3, 224, 224)
        6 视图:  (B, 6, 3, 224, 224)

    Forward 流程：
        1. 所有视图共享 backbone → 各自 (2048,) 特征
        2. 若 symmetric_fusion: 配对 max-pool → (3, 2048) → concat → (6144,)
           否则: 直接 concat → (V*2048,)
        3. 分类头 → (40,) logits
    """

    def __init__(self,
                 backbone: str = "resnet50",
                 num_views: int = 3,
                 symmetric_fusion: bool = False,
                 pretrained: bool = True,
                 dropout: float = 0.5,
                 num_classes: int = NUM_CLASSES):
        super().__init__()

        assert num_views in (3, 6), "num_views 仅支持 3 或 6"
        if num_views == 6:
            assert symmetric_fusion, "6 视图建议启用 symmetric_fusion 以避免特征维度爆炸"

        out_ch, _ = _BACKBONES[backbone]
        self.backbone = _make_backbone(backbone, pretrained=pretrained)
        self.backbone_out_channels = out_ch
        self.num_views = num_views
        self.symmetric_fusion = symmetric_fusion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 融合后特征维度：
        #   symmetric_fusion=True: 3 * out_ch   (always 3 fused views)
        #   symmetric_fusion=False: num_views * out_ch
        if symmetric_fusion:
            fused_dim = 3 * out_ch
        else:
            fused_dim = num_views * out_ch

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
            x: (B, V, 3, H, W) 其中 V = num_views
        Returns:
            logits: (B, num_classes)
        """
        B, V, C, H, W = x.shape
        # 视图合并到 batch 维度共享 backbone
        x_flat = x.view(B * V, C, H, W)                 # (B*V, 3, H, W)

        features = self.backbone(x_flat)                 # (B*V, C_out, 7, 7)
        features = self.avgpool(features)                # (B*V, C_out, 1, 1)
        features = features.view(B * V, -1)              # (B*V, C_out)
        features = features.view(B, V, -1)               # (B, V, C_out)

        if self.symmetric_fusion:
            # 对称融合：相邻两视图 (i, i+1) 是一对，取 element-wise max
            # 6 视图: pairs = [(0,1), (2,3), (4,5)]
            # 对应: xy(+z,-z), xz(+y,-y), yz(+x,-x)
            f1 = features[:, 0::2, :]                    # (B, V/2, C_out)
            f2 = features[:, 1::2, :]                    # (B, V/2, C_out)
            fused = torch.maximum(f1, f2)                # (B, 3, C_out)
            fused = fused.view(B, -1)                    # (B, 3*C_out)
        else:
            fused = features.view(B, -1)                 # (B, V*C_out)

        logits = self.classifier(fused)
        return logits


# ══════════════════════════════════════════════════════════════
#  V3: 置信度门控 + 独立视角分类 + 融合前投影降维
# ══════════════════════════════════════════════════════════════

class MultiViewResNetV3(nn.Module):
    """
    V3: 双重路径多视角分类器。

    设计思路：
      Path A — 传统融合: 每个融合后视角的特征 → 各自线性投影(降维) → concat → 分类
      Path B — 独立视角决策: 每个融合后视角的特征
               → [共享投影 → sigmoid → 置信度 g_i]
               → [独立分类头 → logits_i]
               贡献: g_i × logits_i
      最终输出 = fusion_logits + sum(g_i × view_logits_i)

    为什么这样设计：
      - 每个视角的"独立投票"能力: 正面看和侧面看都应该能识别物体
      - 置信度门控: 不强制每个视角都有用——模型自己学会"这个视角能看清吗"
      - 共享门控投影: "是否可信"是视角无关的通用能力，共享权重合理
      - 融合前降维: 方向已区分，各视角独立投影到低维，减少参数量

    参数：
        backbone:           "resnet18" | "resnet50"
        num_views:          3 或 6（输入视图数）
        symmetric_fusion:   是否 max-pool 对称视图对（6→3）
        dim_reduce:         每视角 projection 降维目标（None=不降维）
        confidence_bias:    置信度投影的 bias 初始化值
        pretrained:         ImageNet 预训练
        dropout:            Dropout 比率
    """

    def __init__(self,
                 backbone: str = "resnet18",
                 num_views: int = 6,
                 symmetric_fusion: bool = True,
                 dim_reduce: int = 256,
                 confidence_bias: float = -1.0,
                 pretrained: bool = True,
                 dropout: float = 0.5,
                 num_classes: int = NUM_CLASSES):
        super().__init__()

        assert num_views in (3, 6), "num_views 仅支持 3 或 6"

        # ── 骨干网络 ──
        C_out, _ = _BACKBONES[backbone]
        self.backbone = _make_backbone(backbone, pretrained=pretrained)
        self.backbone_out_channels = C_out
        self.num_views = num_views
        self.symmetric_fusion = symmetric_fusion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 融合后视角数
        self.num_fused = 3 if symmetric_fusion else num_views

        # 降维目标（不降维则保持原维度）
        reduction = dim_reduce if dim_reduce else C_out

        # ═══════════════════════════════════════════════════════
        #  Path A: 融合前各自线性投影（不共享，方向已区分）
        # ═══════════════════════════════════════════════════════
        self.view_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(C_out, reduction),
                nn.BatchNorm1d(reduction),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.num_fused)
        ])

        # 融合分类头
        fusion_in = self.num_fused * reduction
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # ═══════════════════════════════════════════════════════
        #  Path B: 置信度门控的独立视角分类
        # ═══════════════════════════════════════════════════════

        # 共享置信度投影: 所有视角共用同一个"是否可信"的判断标准
        self.confidence_proj = nn.Linear(C_out, 1)
        nn.init.constant_(self.confidence_proj.bias, confidence_bias)
        # 初始 bias < 0 → sigmoid 偏低 → 初始阶段以融合路径为主
        # 训练中模型自会学会何时提升某个视角的置信度

        # 每视角独立的分类头: 不同视角看到不同的判别特征
        self.view_classifiers = nn.ModuleList([
            nn.Linear(C_out, num_classes)
            for _ in range(self.num_fused)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, V, 3, H, W)  其中 V = num_views
        Returns:
            logits: (B, num_classes)
        """
        B, V, C, H, W = x.shape

        # ── 共享骨干提取 ──
        x_flat = x.view(B * V, C, H, W)
        features = self.backbone(x_flat)             # (B*V, C_out, 7, 7)
        features = self.avgpool(features)            # (B*V, C_out, 1, 1)
        features = features.view(B * V, -1)          # (B*V, C_out)
        features = features.view(B, V, -1)           # (B, V, C_out)

        # ── 对称融合 (6→3) ──
        if self.symmetric_fusion:
            f1 = features[:, 0::2, :]                # (B, 3, C_out)
            f2 = features[:, 1::2, :]
            features = torch.maximum(f1, f2)         # (B, 3, C_out)

        # 此时 features = (B, num_fused, C_out)

        # ════════════════════════════════════════════
        #  Path A: 投影降维 → concat → 融合分类
        # ════════════════════════════════════════════
        projected = []
        for i in range(self.num_fused):
            proj_i = self.view_projections[i](features[:, i, :])
            projected.append(proj_i)
        fused = torch.cat(projected, dim=1)          # (B, num_fused * reduction)
        logits_fusion = self.classifier(fused)       # (B, num_classes)

        # ════════════════════════════════════════════
        #  Path B: 置信度门控 × 独立视角分类
        # ════════════════════════════════════════════
        logits_views = []
        for i in range(self.num_fused):
            feat_i = features[:, i, :]               # (B, C_out)

            # 共享置信度评估: 这个视角的特征"有多可信"
            confidence = torch.sigmoid(
                self.confidence_proj(feat_i)
            )                                        # (B, 1)

            # 独立分类: 从这个视角看，是什么物体
            view_logits = self.view_classifiers[i](feat_i)  # (B, num_classes)

            # 置信度加权贡献
            logits_views.append(confidence * view_logits)

        # 所有视角贡献求和
        logits_views_sum = torch.stack(logits_views, dim=1).sum(dim=1)

        # ════════════════════════════════════════════
        #  最终: 融合路径 + 独立视角贡献
        # ════════════════════════════════════════════
        logits = logits_fusion + logits_views_sum

        return logits


# ══════════════════════════════════════════════════════════════
#  V4: TinyCNN + 十二面体 6 视图 + 5 通道投影
# ══════════════════════════════════════════════════════════════

class TinyCNN(nn.Module):
    """
    手写轻量 CNN 特征提取器，无预训练，适配 56×56×5 投影图。

    结构 (base_ch=32, 输出 128 维):
        56×56×5   → Conv(5→32)    + BN + ReLU → MaxPool(2) → 28×28×32
        28×28×32  → Conv(32→64)   + BN + ReLU → MaxPool(2) → 14×14×64
        14×14×64  → Conv(64→128)  + BN + ReLU → MaxPool(2) → 7×7×128
        7×7×128   → Conv(128→128) + BN + ReLU → AdaptiveAvgPool(1) → 128

    总参数 < 100K（base_ch=32 时 ~98K）。
    """

    def __init__(self, in_channels: int = 5, base_ch: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            # Stage 1: 56² → 28²
            nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Stage 2: 28² → 14²
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Stage 3: 14² → 7²
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Stage 4: 7² → (7², 无降分辨率)
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
        )
        self.out_channels = base_ch * 4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class MultiViewResNetV4(nn.Module):
    """
    V4: TinyCNN + 十二面体 6 视图 + 双重路径（同 V3 门控架构）。

    数据流:
        点云 → PCA → 十二面体 6 向投影 (6×5×56×56)
          ↓
        共享 TinyCNN → 每视图 128d 特征
          ↓
        Path A: 各自投影(128→dim_reduce) → concat → 融合分类
        Path B: 共享置信度投影(128→1)×sigmoid × 独立分类头(128→40) → 求和
          ↓
        final = Path A + Path B

    参数:
        in_channels:  投影图通道数 (default 5)
        base_ch:      TinyCNN 基础通道数 (default 32)
        dim_reduce:   融合前各视角降维目标 (default 32)
        confidence_bias: 置信度投影 bias 初始化
        dropout:      分类头 Dropout
        num_classes:  类别数
    """

    def __init__(self,
                 in_channels: int = 5,
                 base_ch: int = 32,
                 dim_reduce: int = 32,
                 confidence_bias: float = -1.0,
                 dropout: float = 0.5,
                 num_classes: int = NUM_CLASSES):
        super().__init__()

        # ── 骨干 ──
        self.backbone = TinyCNN(in_channels=in_channels, base_ch=base_ch)
        C_out = self.backbone.out_channels  # base_ch * 4
        self.num_views = 6                  # 十二面体 6 个单方向
        self.avgpool = nn.Identity()  # TinyCNN 已经自带 pool

        # ── Path A: 各自投影降维 ──
        reduction = dim_reduce
        self.view_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(C_out, reduction),
                nn.BatchNorm1d(reduction),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.num_views)
        ])

        # 融合分类头
        fusion_in = self.num_views * reduction
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, reduction * 2),
            nn.BatchNorm1d(reduction * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(reduction * 2, num_classes),
        )

        # ── Path B: 置信度门控独立分类 ──
        self.confidence_proj = nn.Linear(C_out, 1)
        nn.init.constant_(self.confidence_proj.bias, confidence_bias)

        self.view_classifiers = nn.ModuleList([
            nn.Linear(C_out, num_classes)
            for _ in range(self.num_views)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 6, 5, 56, 56)  — 十二面体 6 视角 × 5 通道
        Returns:
            logits: (B, num_classes)
        """
        B, V, C, H, W = x.shape

        # ── 共享骨干 ──
        x_flat = x.view(B * V, C, H, W)
        features = self.backbone(x_flat)      # (B*6, C_out)
        features = features.view(B, V, -1)    # (B, 6, C_out)

        # ════════════════════════════════════════════
        #  Path A: 投影降维 → concat → 融合分类
        # ════════════════════════════════════════════
        projected = [self.view_projections[i](features[:, i, :])
                     for i in range(V)]
        fused = torch.cat(projected, dim=1)
        logits_fusion = self.classifier(fused)

        # ════════════════════════════════════════════
        #  Path B: 置信度门控 × 独立视角
        # ════════════════════════════════════════════
        logits_views = []
        for i in range(V):
            feat_i = features[:, i, :]
            confidence = torch.sigmoid(self.confidence_proj(feat_i))
            view_logits = self.view_classifiers[i](feat_i)
            logits_views.append(confidence * view_logits)

        logits_views_sum = torch.stack(logits_views, dim=1).sum(dim=1)

        return logits_fusion + logits_views_sum


# ══════════════════════════════════════════════════════════════
#  V6: V3 + PointNet 交叉注意力路径
# ══════════════════════════════════════════════════════════════

class PointAttentionPath(nn.Module):
    """
    点云自注意力 + PointNet → 与 V3 视角特征做交叉注意力。

    流程:
        原始点云 (B, 2048, 3)
          │
          ▼ 自注意力 (无 embedding，直接在坐标上做)
        (B, 2048, d_attn)
          │
          ▼ PointNet (per-point MLP)
        (B, 2048, d_pointnet)
          │
          ├─→ K: Linear(d_pointnet → d_kv)  → (B, 2048, d_kv)
          └─→ V: Linear(d_pointnet → d_kv)  → (B, 2048, d_kv)
                                               │
        V3 的 3 个融合视角特征 (B, 3, C_out_3v)  │
          │                                     │
          ▼ Q: Linear(C_out_3v → d_kv) → (B, 3, d_kv)
                                               │
          └────────── 交叉注意力 ────────────────┘
                      Q · K^T / √d_kv → softmax
                      weighted sum of V
                    (B, 3, d_kv) → mean pool → (B, d_kv)
                      │
                      ▼ Linear(d_kv → num_classes)
                    logits_pointnet (B, 40)

    参数量 ~800K，占 V3 backbone (12M) 的 ~7%。
    """

    def __init__(self,
                 point_dim: int = 3,
                 d_attn: int = 128,
                 d_pointnet: int = 512,
                 d_kv: int = 128,
                 n_sample: int = 512,
                 c_out_v3: int = 512,
                 num_heads: int = 8,
                 num_classes: int = NUM_CLASSES):
        super().__init__()

        self.d_attn = d_attn
        self.d_kv = d_kv
        self.n_sample = n_sample
        self.num_heads = num_heads
        self.d_head = d_attn // num_heads

        # ── 自注意力 QKV 投影 (直接在坐标上) ──
        self.sa_q = nn.Linear(point_dim, d_attn)
        self.sa_k = nn.Linear(point_dim, d_attn)
        self.sa_v = nn.Linear(point_dim, d_attn)
        self.sa_norm = nn.LayerNorm(d_attn)

        # ── PointNet (per-point MLP, 用 Conv1d, 5 层) ──
        self.point_mlp = nn.Sequential(
            nn.Conv1d(d_attn, 128, 1),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, d_pointnet, 1),
            nn.BatchNorm1d(d_pointnet), nn.ReLU(inplace=True),
        )

        # ── K, V 生成 ──
        self.kv_proj = nn.Conv1d(d_pointnet, d_kv * 2, 1)

        # ── Q 投影 (从 V3 视角特征) ──
        self.q_proj = nn.Linear(c_out_v3, d_kv)

        # ── 输出投影 ──
        self.out_proj = nn.Sequential(
            nn.Linear(d_kv, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, points: torch.Tensor,
                v3_view_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points:           (B, N, 3)         原始/PCA 对齐后的点云
            v3_view_features: (B, 3, C_out_v3)  V3 backbone 的 3 个融合后视角特征

        Returns:
            logits: (B, num_classes)
        """
        B, N, _ = points.shape

        # ════════════════════════════════════════════
        #  0. 随机下采样 → 降低 O(N²) 自注意力显存
        # ════════════════════════════════════════════
        n = min(N, self.n_sample)
        if self.training or n < N:
            idx = torch.randperm(N, device=points.device)[:n]
            idx = idx.unsqueeze(0).expand(B, -1)  # (B, n)
            points = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        N_sa = points.shape[1]

        # ════════════════════════════════════════════
        #  1. 自注意力 (无 learnable embedding)
        # ════════════════════════════════════════════
        q = self.sa_q(points)  # (B, N_sa, d_attn)
        k = self.sa_k(points)
        v = self.sa_v(points)

        # 多头部 reshape
        q = q.view(B, N_sa, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N_sa, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N_sa, self.num_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        sa_out = (attn @ v).transpose(1, 2).contiguous().view(B, N_sa, -1)

        # 残差 + LayerNorm
        sa_out = self.sa_norm(sa_out + self.sa_v(points))  # (B, N_sa, d_attn)

        # ════════════════════════════════════════════
        #  2. PointNet
        # ════════════════════════════════════════════
        pn_in = sa_out.transpose(1, 2)          # (B, d_attn, N)
        pn_out = self.point_mlp(pn_in)           # (B, d_pointnet, N)

        # ════════════════════════════════════════════
        #  3. 生成 K, V (per-point)
        # ════════════════════════════════════════════
        kv = self.kv_proj(pn_out)                # (B, d_kv*2, N)
        k_pn = kv[:, :self.d_kv, :]              # (B, d_kv, N)
        v_pn = kv[:, self.d_kv:, :]              # (B, d_kv, N)

        # ════════════════════════════════════════════
        #  4. Q 来自 V3 的 3 个视角特征
        # ════════════════════════════════════════════
        q_v3 = self.q_proj(v3_view_features)      # (B, 3, d_kv)

        # ════════════════════════════════════════════
        #  5. 交叉注意力: Q_v3(B,3,d) × K_pn(B,N,d)^T
        # ════════════════════════════════════════════
        # scores: (B, 3, N)
        scores = torch.bmm(q_v3, k_pn) / (self.d_kv ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # weighted V: (B, 3, N) × (B, N, d_kv) → (B, 3, d_kv)
        weighted_v = torch.bmm(attn_weights, v_pn.transpose(1, 2))

        # 视角池化 → (B, d_kv)
        pooled = weighted_v.mean(dim=1)

        # ════════════════════════════════════════════
        #  6. 输出投影
        # ════════════════════════════════════════════
        logits = self.out_proj(pooled)             # (B, num_classes)
        return logits


class MultiViewResNetV6(nn.Module):
    """
    V6: V3 架构 + PointNet 交叉注意力路径。

    数据流全景:
        ┌─── 投影视图 ───→ V3 backbone (ResNet18) ───→ 对称融合
        │                                                     │
        │                           V3 路径 (同 v3)            ├─→ 3 个融合视角特征 (B, 3, 512)
        │                         投影 + 门控 + 分类            │         │
        │                              │                       │         ▼ Q
        │                              ▼                       │   交叉注意力 ← K, V ← PointNet
        │                        logits_v3 (B, 40)              │         │
        │                              │                       │         ▼
        │                              │                  logits_pn (B, 40)
        │                              │                       │
        └──────────────────────────────┼───────────────────────┘
                                       ▼
                               final = v3 + pn

    输入格式:
        model((views, points))
        - views:  (B, 6, 3, 224, 224)  六视图投影图
        - points: (B, 2048, 3)          PCA 对齐后的原始点云
    """

    def __init__(self,
                 backbone: str = "resnet18",
                 num_views: int = 6,
                 symmetric_fusion: bool = True,
                 dim_reduce: int = 256,
                 d_attn: int = 128,
                 d_pointnet: int = 512,
                 d_kv: int = 128,
                 n_sample: int = 512,
                 confidence_bias: float = -1.0,
                 pretrained: bool = True,
                 dropout: float = 0.5,
                 num_classes: int = NUM_CLASSES):
        super().__init__()

        # ════════════════════════════════════════════
        #  V3 路径 (完整复用)
        # ════════════════════════════════════════════
        C_out, _ = _BACKBONES[backbone]
        self.backbone = _make_backbone(backbone, pretrained=pretrained)
        self.backbone_out_channels = C_out
        self.num_views = num_views
        self.symmetric_fusion = symmetric_fusion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_fused = 3 if symmetric_fusion else num_views

        reduction = dim_reduce
        self.view_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(C_out, reduction),
                nn.BatchNorm1d(reduction),
                nn.ReLU(inplace=True),
            )
            for _ in range(self.num_fused)
        ])

        fusion_in = self.num_fused * reduction
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.confidence_proj = nn.Linear(C_out, 1)
        nn.init.constant_(self.confidence_proj.bias, confidence_bias)

        self.view_classifiers = nn.ModuleList([
            nn.Linear(C_out, num_classes)
            for _ in range(self.num_fused)
        ])

        # ════════════════════════════════════════════
        #  PointNet 交叉注意力路径
        # ════════════════════════════════════════════
        self.point_path = PointAttentionPath(
            point_dim=3,
            d_attn=d_attn,
            d_pointnet=d_pointnet,
            d_kv=d_kv,
            n_sample=n_sample,
            c_out_v3=C_out,
            num_classes=num_classes,
        )

    def forward(self, x):
        """
        Args:
            x: tuple of (views, points)
               views:  (B, V, C_img, H, W)  投影图
               points: (B, N, 3)            PCA 对齐点云

        Returns:
            logits: (B, num_classes)
        """
        views, points = x
        B, V, C_img, H, W = views.shape

        # ════════════════════════════════════════════
        #  V3 backbone: 视图特征提取
        # ════════════════════════════════════════════
        x_flat = views.view(B * V, C_img, H, W)
        features = self.backbone(x_flat)          # (B*V, C_out, 7, 7)
        features = self.avgpool(features)          # (B*V, C_out, 1, 1)
        features = features.view(B * V, -1)        # (B*V, C_out)
        features = features.view(B, V, -1)         # (B, V, C_out)

        # 对称融合
        if self.symmetric_fusion:
            f1 = features[:, 0::2, :]
            f2 = features[:, 1::2, :]
            features = torch.maximum(f1, f2)       # (B, 3, C_out)

        # ★ 保存融合后视角特征，传给 PointNet 路径做 Q
        v3_view_features = features                 # (B, 3, C_out)

        # ════════════════════════════════════════════
        #  V3 Path A: 投影降维 → concat → 融合分类
        # ════════════════════════════════════════════
        projected = [self.view_projections[i](features[:, i, :])
                     for i in range(self.num_fused)]
        fused = torch.cat(projected, dim=1)
        logits_fusion = self.classifier(fused)

        # ════════════════════════════════════════════
        #  V3 Path B: 置信度门控
        # ════════════════════════════════════════════
        logits_views = []
        for i in range(self.num_fused):
            feat_i = features[:, i, :]
            confidence = torch.sigmoid(self.confidence_proj(feat_i))
            view_logits = self.view_classifiers[i](feat_i)
            logits_views.append(confidence * view_logits)
        logits_views_sum = torch.stack(logits_views, dim=1).sum(dim=1)

        logits_v3 = logits_fusion + logits_views_sum   # (B, 40)

        # ════════════════════════════════════════════
        #  PointNet 交叉注意力路径
        # ════════════════════════════════════════════
        logits_pn = self.point_path(points, v3_view_features)  # (B, 40)

        return logits_v3 + logits_pn
