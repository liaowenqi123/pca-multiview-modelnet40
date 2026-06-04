# PCA-Aligned Multi-View ResNet for 3D Object Classification

A simple yet effective approach for point cloud classification on ModelNet40: align each object via PCA, rasterize orthogonal projections into multi-channel images, then fuse features with a shared ResNet50 backbone.

## Method

1. **PCA Alignment** — Center and rotate each point cloud so its principal axes align with the coordinate frame. This normalizes pose across instances and reduces intra-class variance.
2. **Multi-View Projection** — Three orthogonal views (xy, xz, yz) are rendered as 224×224 pixel images. Each view carries 3 channels: point density, max depth, and min depth.
3. **Shared ResNet50 Backbone** — All three views share the same feature extractor (ImageNet pre-trained), producing a 2048-dim descriptor per view.
4. **Fusion + Classification** — Concatenate the three descriptors into a 6144-dim vector, then pass through a lightweight MLP head (512→256→40).

<p align="center">
  <img src="assets/overview.png" alt="pipeline" width="700"/>
</p>

## Results on ModelNet40 (80/20 split)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|------:|-----------:|----------:|---------:|--------:|
| 1     | 2.207      | 52.5%     | 1.680    | 67.2%   |
| 2     | 1.611      | 70.0%     | 1.402    | 75.1%   |
| 3     | 1.453      | 75.2%     | 1.378    | 75.2%   |
| 4     | 1.382      | 77.3%     | 1.283    | 79.5%   |
| 5     | 1.306      | 80.6%     | 1.229    | **81.6%** |

Best validation accuracy: **81.62%** after 5 epochs (training was still improving when stopped).

## Data Format

Point clouds are stored as `.npy` arrays:
- `train_points.npy`: shape `(N, 2048, 3)`, float32, coordinates in [-1, 1]
- `train_labels.npy`: shape `(N,)`, int32, labels in [0, 39]

## Quick Start

```bash
# Install dependencies
pip install torch torchvision numpy tqdm

# Train
python train.py --epochs 100 --batch_size 64 --lr 0.001
```

## Key Design Choices

- **PCA before projection** — Unlike standard multi-view methods that use fixed camera rigs, PCA alignment adapts the viewing angle to each object's intrinsic shape.
- **Depth-aware channels** — Beyond simple binary/density projection, max/min depth channels encode 3D structure that a plain silhouette would lose.
- **Parameter-efficient sharing** — A single ResNet50 extracts features from all three views, keeping the model compact (~25M backbone + ~3M classifier).

## Files

| File | Description |
|------|-------------|
| `train.py` | Training script with model definition, data loading, and full pipeline |
| `categories.txt` | 40 class names for ModelNet40 |

## License

MIT
