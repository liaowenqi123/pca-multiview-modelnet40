# V7: Frozen ResNet + PointNet Latent Self-Attention + Gated Fusion

## 最终结果

**ModelNet40 测试集准确率: 84%**（提交到线上平台）

## 架构概述

V7 = 两条并行路径 + 一个可学习门控融合器。

```
        ┌──────── V3 路径 (多视图 CNN) ────────→ f₁ ∈ ℝ^(B×40)
        │
输入 ───┤
        │
        └──────── PointNet 路径 (点云) ──────────→ f₂ ∈ ℝ^(B×40)
                                                         │
                                              FusionGate(f₁, f₂) → logits
```

---

## 1. 数据预处理

### 1.1 PCA 对齐

给定点云 $P \in \mathbb{R}^{N \times 3}$ (N=2048)：

$$\bar{P} = \frac{1}{N}\sum_i P_i$$

$$\Sigma = \frac{1}{N}(P - \bar{P})^\top(P - \bar{P}) \in \mathbb{R}^{3 \times 3}$$

特征分解 $\Sigma = V \Lambda V^\top$，其中 $\Lambda = \text{diag}(\lambda_1, \lambda_2, \lambda_3)$，$\lambda_1 \geq \lambda_2 \geq \lambda_3$。

$$P_{aligned} = (P - \bar{P}) \cdot V$$

方向消歧：若 $\text{median}(P_{aligned}[:, i]) < 0$，则 $P_{aligned}[:, i] \leftarrow -P_{aligned}[:, i]$

归一化：$P_{aligned} \leftarrow P_{aligned} / \max(|P_{aligned}|)$，使坐标 $\in [-1, 1]$

### 1.2 多视图投影

对对齐后的点云做 6 视图投影（正交三平面 × 正反方向），每视图 3 通道：

$$\text{view}^{(k)} = \begin{bmatrix} \text{density} \\ \text{max\_depth} \\ \text{min\_depth} \end{bmatrix} \in \mathbb{R}^{3 \times 224 \times 224}$$

投影细节：坐标 $[u, v] \in [-1, 1]$ 映射到 $[0, 223]$ 整数网格，每个 pixel 统计落在该位置的点的密度（归一化）、最大/最小深度值。

6 视图顺序：$xy(+z), xy(-z), xz(+y), xz(-y), yz(+x), yz(-x)$

### 1.3 数据增强（仅训练时）

$$P_{aug} = (P_{aligned} \cdot R_z(\theta)) \cdot s + t + \epsilon$$

其中 $R_z(\theta)$ 是绕 z 轴旋转矩阵，$\theta \sim U(0, 2\pi)$，$s \sim U(0.9, 1.1)$，$t \sim U(-0.05, 0.05)$，$\epsilon \sim \mathcal{N}(0, 0.01)$。之后重新归一化到 $[-1, 1]$。

---

## 2. V3 路径：多视图 CNN

### 2.1 特征提取

6 视图张量 $X \in \mathbb{R}^{B \times 6 \times 3 \times 224 \times 224}$ 展平到 batch 维度后过共享 ResNet18：

$$h_i = \text{ResNet18}(X_i) \in \mathbb{R}^{512}, \quad i = 1, \dots, 6B$$

恢复后 $H \in \mathbb{R}^{B \times 6 \times 512}$

**冻结策略**：ResNet18 被包装为 `nn.Sequential`（`[conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]`）。默认冻结 conv1, bn1, layer1, layer2（索引 0,1,4,5），只训练 layer3 和 layer4。冻结参数约 7M，可训练约 5M。

### 2.2 对称融合

相邻视图为对称对（正反方向投影），取逐元素最大值消除 PCA 方向歧义：

$$H^{fused} = \max(H_{[:,0::2,:]}, H_{[:,1::2,:]}) \in \mathbb{R}^{B \times 3 \times 512}$$

三个融合视角记作 $h^{xy}, h^{xz}, h^{yz}$。

### 2.3 双重分类路径

**Path A — 融合分类**：每个融合视角独立投影降维后 concat

$$z_i^A = \text{ReLU}(\text{BN}(W_i^A h_i + b_i^A)), \quad W_i^A \in \mathbb{R}^{512 \times 256}$$

$$z_{cat} = [z_0^A \| z_1^A \| z_2^A] \in \mathbb{R}^{768}$$

$$\text{logits}_{fus} = \text{MLP}(z_{cat}), \quad 768 \to 512 \to 256 \to 40$$

**Path B — 置信度门控**：每个视角独立分类 + 共享置信度

$$g_i = \sigma(W_g h_i + b_g), \quad W_g \in \mathbb{R}^{512 \times 1} \text{ (共享)}$$

$$\text{logits}_i^B = W_i^B h_i + b_i^B, \quad W_i^B \in \mathbb{R}^{512 \times 40}$$

$$\text{logits}_{gate} = \sum_{i=1}^3 g_i \cdot \text{logits}_i^B$$

**V3 路径输出**：

$$f_1 = \text{logits}_{fus} + \text{logits}_{gate} \in \mathbb{R}^{B \times 40}$$

---

## 3. PointNet 路径

### 3.1 随机下采样

为控制自注意力 $O(N^2)$ 显存，随机采样 $M=512$ 个点：

$$P' = \text{RandSample}(P_{aligned}, M) \in \mathbb{R}^{B \times 512 \times 3}$$

训练时每轮随机不同子集，天然数据增强。

### 3.2 Compress: 坐标 → 低维 latent

用 Conv1d 将坐标压缩到低维空间：

$$h_{comp} = \text{ReLU}(\text{BN}(\text{Conv1d}_{32}(\text{ReLU}(\text{BN}(\text{Conv1d}_3(P')))))) \in \mathbb{R}^{B \times 64 \times 512}$$

即 $3 \to 32 \to 64$，记 $d_l = 64$。

### 3.3 自注意力（在 latent 空间）

转置得 $H \in \mathbb{R}^{B \times 512 \times 64}$，single-head 自注意力：

$$Q = H W_q, \quad K = H W_k, \quad V = H W_v, \quad W_{q,k,v} \in \mathbb{R}^{64 \times 64}$$

$$A = \text{softmax}\left(\frac{Q K^\top}{\sqrt{64}}\right) \in \mathbb{R}^{B \times 512 \times 512}$$

$$H_{sa} = \text{LayerNorm}(A V + H)$$

与 V6 的关键区别：自注意力在 PointNet 压缩后的 64d latent 空间进行，而非原始 3d 坐标。语义更丰富，参数量可控。

### 3.4 Expand: latent → 高维点特征

$$\begin{aligned}
h_1 &= \text{ReLU}(\text{BN}(\text{Conv1d}_{128}(H_{sa}^\top))) \\
h_2 &= \text{ReLU}(\text{BN}(\text{Conv1d}_{256}(h_1))) \\
h_{pn} &= \text{ReLU}(\text{BN}(\text{Conv1d}_{512}(h_2))) \in \mathbb{R}^{B \times 512 \times 512}
\end{aligned}$$

即 $64 \to 128 \to 256 \to 512$。

### 3.5 交叉注意力：视角 ↔ 点

**Query 来源**：从 V3 的三个融合视角特征 $h^{xy}, h^{xz}, h^{yz}$ 投影

$$Q_{v3} = W_q h^{fused} \in \mathbb{R}^{B \times 3 \times 128}, \quad W_q \in \mathbb{R}^{512 \times 128}$$

**Key/Value 来源**：从 PointNet 每点特征生成

$$K_{pn} = \text{Conv1d}_{k}(h_{pn}) \in \mathbb{R}^{B \times 128 \times 512}$$
$$V_{pn} = \text{Conv1d}_{v}(h_{pn}) \in \mathbb{R}^{B \times 128 \times 512}$$

**交叉注意力**：

$$\text{scores} = \frac{Q_{v3} \cdot K_{pn}}{\sqrt{128}} \in \mathbb{R}^{B \times 3 \times 512}$$

$$A_{cross} = \text{softmax}(\text{scores}, \dim=-1)$$

$$\text{weighted} = A_{cross} \cdot V_{pn}^\top \in \mathbb{R}^{B \times 3 \times 128}$$

**视角池化 + 输出**：

$$h_{out} = \frac{1}{3}\sum_{i=1}^3 \text{weighted}_i \in \mathbb{R}^{B \times 128}$$

$$f_2 = \text{MLP}(h_{out}), \quad 128 \to 256 \to 128 \to 40$$

---

## 4. FusionGate：可学习门控融合

两路 logits $f_1, f_2 \in \mathbb{R}^{B \times 40}$ 通过可学习门控融合：

### 乘法项

$$\text{mult} = \sigma(f_1) \odot \sigma(f_2)$$

逐类独立 sigmoid 后逐元素相乘。直观含义：只有两路都认为某类高概率时，该类才得分。

### 加法项

$$\text{add} = f_1 + f_2$$

标准 logit 相加，保证容错。

### 可学习门控

$$w = \sigma(w_{raw}), \quad w_{raw} \in \mathbb{R} \text{ (可学习)}$$

$$\text{score} = w \cdot \text{mult} + (1 - w) \cdot \text{add}$$

初始 $w_{raw} = 0$，即 $w = 0.5$，两路等权。

### 可学习温度

$$T = \exp(t_{raw}), \quad t_{raw} \in \mathbb{R} \text{ (可学习)}$$

$$\text{logits} = \text{score} \cdot T$$

初始 $T = 1.0$，训练中模型自适应调整置信度锐度。

最终优化目标为 CrossEntropyLoss（label smoothing 0.1）：

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^B \sum_{c=1}^{40} \left[(1 - 0.1) \cdot y_{i,c} + \frac{0.1}{39}\right] \cdot \log \text{softmax}(\text{logits}_{i,c})$$

---

## 5. 优化器配置

| 参数组 | lr 倍率 | 说明 |
|--------|---------|------|
| backbone (layer3-4) | 0.1× | 预训练权重微调 |
| view_projections | 0.1× | 投影层，需稳定 |
| classifier | 1× | 主要学习目标 |
| confidence_proj | 1× | 门控置信度 |
| view_classifiers | 1× | 独立视角分类 |
| point_path | 1× | PointNet 全路径 |
| fusion_gate | 0.01× | 融合权重变化需缓慢 |

AdamW optimizer, weight_decay=1e-4, CosineAnnealingLR (T_max=epochs, eta_min=lr×0.01)

---

## 6. 参数量

| 模块 | 参数 |
|------|------|
| ResNet18 backbone (layer1-2 冻结) | ~7M (冻结) |
| ResNet18 backbone (layer3-4 可训) | ~5M |
| view_projections + classifier | ~0.8M |
| confidence_proj + view_classifiers | ~0.1M |
| PointNetPathV7 (compress+SA+expand+cross-attn+out) | ~0.5M |
| FusionGate | 2 |
| **总可训练** | **~6.4M** |

---

## 7. 关键设计决策

| 决策 | 理由 |
|------|------|
| 冻结 layer1-2 | 底层边缘/纹理特征通用，冻结减少过拟合 |
| SA 在 PointNet 压缩后 | 64d latent 空间做 attention 比原始 3d 坐标更有语义，同时控制参数量 |
| 单头自注意力 | 64d 用多头会分得太碎，单头 + LayerNorm 足够 |
| 乘法融合项 | 要求两路一致决策 → 去噪，减少单路错误传播 |
| 可学习温度 | 自适应调节分类置信度锐度 |
| 融合 gate lr=0.01× | 门控权重变化太快会破坏训练稳定性 |

## 运行

```bash
# 多文件版本 (import common/)
python v7_fused_gate/train.py --epochs 100 --batch_size 64

# 自包含版本 (单文件)
python v7_fused_gate/full_train.py --epochs 100 --batch_size 64
```
