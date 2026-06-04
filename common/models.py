"""
MultiViewResNet：多视角 ResNet 特征提取 + 对称融合 + 分类。
"""

import torch
import torch.nn as nn

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
