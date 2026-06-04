# V2: ResNet18 + 6-View Symmetric Fusion

## 一句话概括

在 v1 基础上解决两个问题：(1) ResNet50→ResNet18 降参数防过拟合，(2) 每正交面投正反两个方向 + max-pool 融合，消除 PCA 方向歧义。

## 核心改进

### PCA 方向歧义如何被解决？

PCA 对一个杯子取向的结果可能是方向 A，对另一个可能是方向 B（median 定向不可靠）。六视图的做法是：**每个正交轴都从正反两个方向各投影一次，然后 max-pool 取最强响应**。

```
物体 A: PCA → +z 朝上    物体 B: PCA → +z 朝下

A 的 xy(+z): 看到杯口        B 的 xy(+z): 看到杯底
A 的 xy(-z): 看到杯底        B 的 xy(-z): 看到杯口

max(杯口, 杯底) = 杯口 → 两个物体得到相同的融合特征 ✓
```

## 数据流向

```
原始点云 (2048, 3)
    │
    ▼ PCA 对齐
对齐点云 (2048, 3), 坐标 ∈ [-1, 1]
    │
    ▼ 六视图投影
    ├─ xy(+z): 从 +z 看 xy 平面  → (3, 224, 224)  [正]
    ├─ xy(-z): 从 -z 看 xy 平面  → (3, 224, 224)  [反]  ─┐
    │                                                       │ 配对
    ├─ xz(+y): 从 +y 看 xz 平面  → (3, 224, 224)  [正]    │
    ├─ xz(-y): 从 -y 看 xz 平面  → (3, 224, 224)  [反]  ─┤
    │                                                       │ 配对
    ├─ yz(+x): 从 +x 看 yz 平面  → (3, 224, 224)  [正]    │
    └─ yz(-x): 从 -x 看 yz 平面  → (3, 224, 224)  [反]  ─┘
    │
    ▼ 堆叠 + ImageNet 归一化
输入张量 (B, 6, 3, 224, 224)
    │
    ▼ 展开到 batch 维度
(B*6, 3, 224, 224)
    │
    ▼ 共享 ResNet18 (无最后 avgpool/FC)
    ├─ conv1 (7×7, 64)
    ├─ bn1 + ReLU + MaxPool
    ├─ layer1 (2 BasicBlock,  64c)
    ├─ layer2 (2 BasicBlock, 128c)
    ├─ layer3 (2 BasicBlock, 256c)
    └─ layer4 (2 BasicBlock, 512c)
(B*6, 512, 7, 7)
    │
    ▼ AdaptiveAvgPool2d(1)
(B*6, 512)
    │
    ▼ 恢复视图维度
(B, 6, 512)
    │
    ▼ 对称 max-pool: max(奇, 偶)  ← 三对对称视图融合
(B, 3, 512)
    │
    ▼ concat
(B, 3×512) = (B, 1536)
    │
    ▼ 分类头
    Linear(1536 → 512) → BN → ReLU → Dropout(0.5)
    Linear(512  → 256) → BN → ReLU → Dropout(0.5)
    Linear(256  →  40)
    │
    ▼
logits (B, 40)
```

## 对称融合关键代码

```python
# features: (B, 6, 512)
# 顺序: [xy(+z), xy(-z), xz(+y), xz(-y), yz(+x), yz(-x)]
f1 = features[:, 0::2, :]  # 奇数位 = 正方向
f2 = features[:, 1::2, :]  # 偶数位 = 反方向
fused = torch.maximum(f1, f2)  # (B, 3, 512)
```

## 关键数值

| 项目 | v1 | v2 |
|------|----|----|
| Backbone | ResNet50 | ResNet18 |
| 原始视图 | 3 | 6 |
| 融合后视图 | 3 | 3 |
| 每视图特征维 | 2048 | 512 |
| 分类头输入 | 6144 | 1536 |
| 总参数 | ~26.6M | ~11.4M |
| PCA 鲁棒 | 否 | 是 |

## 运行

```bash
python v2_resnet18_6view_sym/train.py --epochs 100 --batch_size 64
```
