"""
ModelNet40 多视图数据集（支持正交投影、十二面体投影）。
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    pca_align, project, augment_points,
    multi_view_project_dodeca,
    IMAGENET_MEAN, IMAGENET_STD,
    NORM_5CH_MEAN, NORM_5CH_STD,
    GRID_SIZE,
)


class ModelNet40MultiView(Dataset):
    """ModelNet40 多视图数据集。"""

    def __init__(self, points: np.ndarray, labels: np.ndarray,
                 num_views: int = 3,
                 projection: str = "ortho",     # "ortho" | "dodeca"
                 grid_size: int = GRID_SIZE,
                 augment: bool = False,
                 cache_pca: bool = True):
        """
        Args:
            points:    (N, 2048, 3) 点云
            labels:    (N,) 标签
            num_views: 视图数 (ortho: 3/6, dodeca: 固定 6)
            projection: "ortho"=正交三/六视图, "dodeca"=十二面体 6 向 5 通道
            grid_size: 投影图像分辨率
            augment:   是否做数据增强
            cache_pca: 是否缓存 PCA 对齐结果
        """
        self.points = points
        self.labels = labels
        self.num_views = num_views
        self.projection = projection
        self.grid_size = grid_size
        self.augment = augment
        self.cache_pca = cache_pca

        # dodeca 模式下固定使用 5 通道归一化
        self._dodeca = (projection == "dodeca")
        if self._dodeca:
            self.norm_mean = NORM_5CH_MEAN
            self.norm_std  = NORM_5CH_STD
        else:
            self.norm_mean = IMAGENET_MEAN
            self.norm_std  = IMAGENET_STD

        if cache_pca:
            mode = f"{num_views}视图, {projection}"
            print(f"[dataset] PCA 对齐 ({len(points)} 个样本, {mode})...")
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
        if self.aligned is not None:
            pts = self.aligned[idx].copy()
        else:
            pts = pca_align(self.points[idx])

        if self.augment:
            pts = augment_points(pts)

        # 多视图投影
        if self._dodeca:
            views = multi_view_project_dodeca(pts, self.grid_size)
        else:
            views = project(pts, num_views=self.num_views, grid_size=self.grid_size)

        # 归一化
        views = (views - self.norm_mean) / self.norm_std

        label = self.labels[idx]
        return torch.from_numpy(views), torch.tensor(label, dtype=torch.long)
