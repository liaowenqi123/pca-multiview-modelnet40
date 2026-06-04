# V1: ResNet50 + 3-View（基线）

## 一句话概括

PCA 对齐点云 → 三个正交面投影为图像 → 共享 ResNet50 提特征 → concat → 分类。

## 数据流向

```
原始点云 (2048, 3)
    │
    ▼ PCA 对齐
对齐点云 (2048, 3), 坐标 ∈ [-1, 1]
    │
    ▼ 三视图投影 — 每个视图 3 通道(密度, max深度, min深度)
    ├─ xy 平面 (z 为深度) → (3, 224, 224)
    ├─ xz 平面 (y 为深度) → (3, 224, 224)
    └─ yz 平面 (x 为深度) → (3, 224, 224)
    │
    ▼ 堆叠 + ImageNet 归一化
输入张量 (B, 3, 3, 224, 224)
    │
    ▼ 展开视图到 batch 维度
(B*3, 3, 224, 224)  — 3 个视图合并成 3 倍 batch
    │
    ▼ 共享 ResNet50 (无最后 avgpool/FC)
    ├─ conv1 (7×7, 64)
    ├─ bn1 + ReLU + MaxPool
    ├─ layer1 (3 Bottleneck, 256c)
    ├─ layer2 (4 Bottleneck, 512c)
    ├─ layer3 (6 Bottleneck, 1024c)
    └─ layer4 (3 Bottleneck, 2048c)
(B*3, 2048, 7, 7)
    │
    ▼ AdaptiveAvgPool2d(1)
(B*3, 2048)
    │
    ▼ 恢复视图维度 + concat
(B, 3×2048) = (B, 6144)
    │
    ▼ 分类头
    Linear(6144 → 512) → BN → ReLU → Dropout(0.5)
    Linear(512  → 256) → BN → ReLU → Dropout(0.5)
    Linear(256  →  40)
    │
    ▼
logits (B, 40)
```

## 关键数值

| 项目 | 值 |
|------|-----|
| Backbone | ResNet50 (ImageNet 预训练) |
| Backbone 参数 | ~23.5M |
| 分类头参数 | ~3.1M |
| 总参数 | ~26.6M |
| 分类头输入 | 6144 (3×2048) |
| Dropout | 0.5 |
| Label Smoothing | 0.1 |

## 已知问题

- **过拟合**: Epoch 9 val=84.2%, epoch 12 val=82.5%, gap 扩大到 5%
- **PCA 方向歧义**: median 定向不可靠，同类物体可能取向相反
- **覆盖不足**: 仅 3 个正交面，倾斜视角的判别信息丢失

## 运行

```bash
# 训练
python v1_resnet50_3view/train.py --epochs 100 --batch_size 64

# 推理
python v1_resnet50_3view/inference.py
```
