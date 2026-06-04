# PCA-Aligned Multi-View ResNet for 3D Object Classification

A simple yet effective approach for point cloud classification on ModelNet40: align each object via PCA, rasterize orthogonal projections into multi-channel images, then fuse features with a shared ResNet backbone.

## Method

1. **PCA Alignment** - Center and rotate each point cloud so its principal axes align with the coordinate frame. Normalizes pose across instances.
2. **Multi-View Projection** - Orthogonal views rendered as 224x224 pixel images, each carrying 3 channels: point density, max depth, and min depth.
3. **Shared ResNet Backbone** - All views share one feature extractor (ImageNet pre-trained), producing compact per-view descriptors.
4. **Fusion + Classification** - View features are fused and passed through a lightweight MLP head (512-256-40).

## Variants

| Folder | Backbone | Views | Fusion | Notes |
|--------|----------|-------|--------|-------|
| `v1_resnet50_3view/` | ResNet50 | 3 (xy, xz, yz) | Concat | Baseline |
| `v2_resnet18_6view_sym/` | ResNet18 | 6 (3 pairs) | Symmetric max-pool | PCA-robust |

### v2: Symmetric Fusion

v2 addresses PCA's inherent axis-sign ambiguity by projecting each orthogonal plane from both directions:

```
xy(+z) / xy(-z) → max-pool → fused xy
xz(+y) / xz(-y) → max-pool → fused xz
yz(+x) / yz(-x) → max-pool → fused yz
```

The paired max-pooling makes the representation invariant to PCA orientation flips, while a lighter ResNet18 backbone reduces overfitting.

## Results

**v1 (ResNet50 + 3-View)** on 80/20 split:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|------:|-----------:|----------:|---------:|--------:|
| 1 | 2.207 | 52.5% | 1.680 | 67.2% |
| 2 | 1.611 | 70.0% | 1.402 | 75.1% |
| 3 | 1.453 | 75.2% | 1.378 | 75.2% |
| 4 | 1.382 | 77.3% | 1.283 | 79.5% |
| 5 | 1.306 | 80.6% | 1.229 | **81.6%** |
| 6 | 1.265 | 81.8% | 1.253 | 81.1% |
| 7 | 1.213 | 83.9% | 1.204 | 81.2% |
| 8 | 1.187 | 84.4% | 1.166 | **83.0%** |
| 9 | 1.159 | 85.6% | 1.154 | **84.2%** |
| 10 | 1.136 | 86.1% | 1.146 | 83.5% |
| 11 | 1.107 | 87.2% | 1.152 | 83.8% |
| 12 | 1.090 | 87.7% | 1.171 | 82.5% |

Peak validation accuracy: **84.15%** (epoch 9). Overfitting visible from epoch 10 onward.

## Data Format

Point clouds as `.npy` arrays:
- `train_points.npy`: `(N, 2048, 3)`, float32
- `train_labels.npy`: `(N,)`, int32, labels 0-39

## Quick Start

```bash
pip install torch torchvision numpy tqdm

# v1: ResNet50 + 3-view
python v1_resnet50_3view/train.py --epochs 100 --batch_size 64

# v2: ResNet18 + 6-view symmetric fusion
python v2_resnet18_6view_sym/train.py --epochs 100 --batch_size 64
```

## Project Structure

```
├── common/                 # Shared modules (preprocessing, models, dataset, training utils)
│   ├── preprocessing.py    # PCA alignment, multi-view projection, augment
│   ├── dataset.py          # ModelNet40MultiView dataset
│   ├── models.py           # MultiViewResNet (configurable backbone & fusion)
│   └── train_utils.py      # train_one_epoch, evaluate
├── v1_resnet50_3view/      # Baseline: ResNet50 + 3 orthogonal views
├── v2_resnet18_6view_sym/  # Improved: ResNet18 + 6 views with symmetric fusion
├── categories.txt          # 40 ModelNet40 class names
└── requirements.txt
```

## License

MIT
