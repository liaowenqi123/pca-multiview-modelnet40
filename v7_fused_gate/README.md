# V7: Frozen Backbone + PointNet Latent SA + Gated Fusion

## 一句话概括

冻结 ResNet 前半部分、PointNet 先压缩再在 latent 空间做自注意力、用可学习门控融合两路 logits。

## 相比 V6 的三个关键改动

| | V6 | V7 |
|---|---|---|
| Backbone | 全参数训练 (12M) | **layer1-2 冻结** (~5M 可训) |
| 自注意力 | 原始坐标上 (3→128) | **PointNet 压缩后 latent (64d)** |
| 两路融合 | `f1 + f2` | **FusionGate** |

## 数据流

```
                   ┌────── V3 Path ──────────┐
                   │ 6 views → ResNet18       │
                   │   layer1-2 ★冻结★        │
                   │   layer3-4 可训练         │
                   │   sym fusion → 3 views   │
                   │   gate + fusion clf      │
                   │          ↓               │
                   │      f1 (B, 40)          ├──→ FusionGate ──→ final
                   │                          │        ↑
                   └── PointNet Latent SA ────┘        │
                       points (512, 3)                  │
                         ↓ compress                     │
                       (512, 64) latent                 │
                         ↓ 自注意力(single-head)         │
                       (512, 64)                        │
                         ↓ expand                       │
                       (512, 512)                       │
                         ↓ K,V ← Q from V3              │
                         ↓ 交叉注意力 → output           │
                              ↓                         │
                          f2 (B, 40) ──────────────────┘
```

## FusionGate

```
f1 = V3 logits (B, 40)
f2 = PointNet logits (B, 40)

multiplicative = σ(f1) ⊙ σ(f2)   # 两路同时激活的类才得分
additive       = f1 + f2          # 标准 logit 相加

w = sigmoid(w_raw)                # 可学习门控权重
score = w * multiplicative + (1-w) * additive

temperature = exp(log_temp)       # 可学习温度
logits = score * temperature
```

**直觉**：乘法端要求两路一致（去噪），加法端保证容错。w 在学习中自动调整平衡点。

纯乘法模式：`python train.py --pure_mul`（去掉 w 门控，只做 sigmoid 乘积）。

## 关键数值

| 项目 | V6 | V7 |
|------|-----|-----|
| Backbone 可训 | 12M | **~5M** |
| PointNet 参数 | ~800K | ~500K |
| 总可训参数 | ~13M | **~6M** |
| 自注意力位置 | 原始坐标 | **latent 压缩后** |
| 自注意力维度 | 128d multi-head | **64d single-head** |
| 融合方式 | f1 + f2 | **FusionGate** |

## 运行

```bash
# 默认（门控融合）
python v7_fused_gate/train.py --epochs 100 --batch_size 64

# 纯乘法融合
python v7_fused_gate/train.py --epochs 100 --pure_mul

# 自定义冻结层
python v7_fused_gate/train.py --freeze_until layer3
```
