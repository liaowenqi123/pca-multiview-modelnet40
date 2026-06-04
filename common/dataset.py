"""
ModelNet40 多视图数据集。
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    pca_align, project, augment_points,
    IMAGENET_MEAN, IMAGENET_STD, GRID_SIZE,
)


class ModelNet40MultiView(Dataset):
    """ModelNet40 多视图数据集，支持 3 视图和 6 视图。"""

    def __init__(self, points: np.ndarray, labels: np.ndarray,
                 num_views: int = 3,
                 augment: bool = False,
                 cache_pca: bool = True):
        """
        Args:
            points: (N, 2048, 3) 点云
            labels: (N,) 标签
            num_views: 3 或 6
            augment: 是否做数据增强（仅训练时）
            cache_pca: 是否缓存 PCA 对齐结果
        """
        self.points = points
        self.labels = labels
        self.num_views = num_views
        self.augment = augment
        self.cache_pca = cache_pca

        if cache_pca:
            print(f"[dataset] PCA 对齐 ({len(points)} 个样本, {num_views} 视图)...")
            self.aligned = []
            for i in range(len(points)):
                self.aligned.append(pca_align(points[i]))
                if (i + 1) % 1000 == 0:
                    print(f"  PCA 进度: {i+1}/{len(points)}")
            print("[dataset] PCA 对齐完成。")
        else:
            self.aligned = None
            print("[dataset] PCA 将在 __getitem__ 中实时计算。")

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        # 获取对齐后点云
        if self.aligned is not None:
            pts = self.aligned[idx].copy()
        else:
            pts = pca_align(self.points[idx])

        # 数据增强
        if self.augment:
            pts = augment_points(pts)

        # 多视图投影
        views = project(pts, num_views=self.num_views, grid_size=GRID_SIZE)

        # ImageNet 归一化
        views = (views - IMAGENET_MEAN) / IMAGENET_STD

        label = self.labels[idx]
        return torch.from_numpy(views), torch.tensor(label, dtype=torch.long)
