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
