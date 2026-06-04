"""
点云预处理：PCA 对齐、三视/六视投影、数据增强。
"""

import numpy as np

# ─── 常量 ────────────────────────────────────────────────────
NUM_CLASSES = 40
NUM_POINTS = 2048
GRID_SIZE = 224          # ResNet 标准输入

# ImageNet 归一化（与 torchvision 预训练模型一致）
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ══════════════════════════════════════════════════════════════
#  PCA 对齐
# ══════════════════════════════════════════════════════════════

def pca_align(points: np.ndarray) -> np.ndarray:
    """
    对点云做 PCA 对齐。

    步骤：
      1. 去中心化（减质心）
      2. 协方差矩阵 (3×3) + 特征分解
      3. 按特征值降序：最大→x，次大→y，最小→z
      4. 旋转点云，保证右手系
      5. 方向模糊性处理（median 定向）
      6. 归一化到 [-1, 1]³

    Args:
        points: (N, 3) 原始点云

    Returns:
        aligned: (N, 3) 对齐归一化后的点云
    """
    N = points.shape[0]

    # 去中心化
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid

    # 协方差矩阵 + 特征分解
    cov = (centered.T @ centered) / N
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # 升序

    # 降序排列
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # 保证右手系
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    # 旋转
    aligned = centered @ eigenvectors

    # 方向模糊性：每轴 majority 朝向正半轴
    for i in range(3):
        if np.median(aligned[:, i]) < 0:
            aligned[:, i] *= -1

    # 归一化到 [-1, 1]
    max_abs = np.max(np.abs(aligned))
    if max_abs > 1e-8:
        aligned = aligned / max_abs

    return aligned


# ══════════════════════════════════════════════════════════════
#  散射投影
# ══════════════════════════════════════════════════════════════

def _scatter_project(points_2d: np.ndarray,
                     values_density: np.ndarray,
                     values_max: np.ndarray,
                     values_min: np.ndarray,
                     grid_size: int = GRID_SIZE,
                     ) -> np.ndarray:
    """
    将 2D 点散射到 grid×grid 网格，生成三通道图像。

    通道 0：密度（归一化点计数）
    通道 1：最大深度值
    通道 2：最小深度值

    Args:
        points_2d: (M, 2) 投影坐标，范围 [-1, 1]
        values_density: (M,) 密度权重
        values_max: (M,) 深度值（max 通道）
        values_min: (M,) 深度值（min 通道）
        grid_size: 分辨率

    Returns:
        image: (3, grid_size, grid_size) float32
    """
    coords = ((points_2d + 1.0) / 2.0 * (grid_size - 1)).astype(np.int32)
    coords = np.clip(coords, 0, grid_size - 1)

    x_idx = coords[:, 0]
    y_idx = coords[:, 1]
    flat_idx = y_idx * grid_size + x_idx

    # 密度
    density = np.zeros(grid_size * grid_size, dtype=np.float32)
    np.add.at(density, flat_idx, values_density.astype(np.float32))

    # 最大深度
    max_depth = np.full(grid_size * grid_size, -np.inf, dtype=np.float32)
    np.maximum.at(max_depth, flat_idx, values_max.astype(np.float32))

    # 最小深度
    min_depth = np.full(grid_size * grid_size, np.inf, dtype=np.float32)
    np.minimum.at(min_depth, flat_idx, values_min.astype(np.float32))

    # reshape
    density = density.reshape(grid_size, grid_size)
    max_depth = max_depth.reshape(grid_size, grid_size)
    min_depth = min_depth.reshape(grid_size, grid_size)

    # 空 cell 填 0
    max_depth[np.isneginf(max_depth)] = 0.0
    min_depth[np.isposinf(min_depth)] = 0.0

    # 归一化
    d_max = density.max()
    if d_max > 0:
        density /= d_max

    max_depth = np.clip((max_depth + 1.0) / 2.0, 0.0, 1.0)
    min_depth = np.clip((min_depth + 1.0) / 2.0, 0.0, 1.0)

    return np.stack([density, max_depth, min_depth], axis=0).astype(np.float32)


# ══════════════════════════════════════════════════════════════
#  多视图投影
# ══════════════════════════════════════════════════════════════

def multi_view_project(points: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    三视图投影：xy (+z 方向)、xz (+y 方向)、yz (+x 方向)。

    Returns:
        views: (3, 3, grid_size, grid_size)
               维度 0 = [xy, xz, yz]
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ones = np.ones(points.shape[0], dtype=np.float32)

    xy_view = _scatter_project(np.stack([x, y], axis=1), ones, z, z, grid_size)
    xz_view = _scatter_project(np.stack([x, z], axis=1), ones, y, y, grid_size)
    yz_view = _scatter_project(np.stack([y, z], axis=1), ones, x, x, grid_size)

    return np.stack([xy_view, xz_view, yz_view], axis=0)


def multi_view_project_6(points: np.ndarray, grid_size: int = GRID_SIZE) -> np.ndarray:
    """
    六视图投影：3 个正交面各投正反两个方向。

    视图顺序（成对排列，便于对称融合）：
      [xy(+z),  xy(-z),   — 从 +z / -z 方向看 xy 平面
       xz(+y),  xz(-y),   — 从 +y / -y 方向看 xz 平面
       yz(+x),  yz(-x)]   — 从 +x / -x 方向看 yz 平面

    对称视图对的关系：
      - 密度通道相同（同一平面的 2D 投影）
      - 深度通道相反（从相反方向观察的深度值）

    这种设计天然处理 PCA 方向符号歧义：无论 PCA 把主轴指向哪边，
    对称 max-pool 后结果不变。

    Returns:
        views: (6, 3, grid_size, grid_size)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    ones = np.ones(points.shape[0], dtype=np.float32)

    # xy 平面 — z 正反两个方向
    xy_pos_z = _scatter_project(np.stack([x, y], axis=1), ones, z, z, grid_size)
    xy_neg_z = _scatter_project(np.stack([x, y], axis=1), ones, -z, -z, grid_size)

    # xz 平面 — y 正反两个方向
    xz_pos_y = _scatter_project(np.stack([x, z], axis=1), ones, y, y, grid_size)
    xz_neg_y = _scatter_project(np.stack([x, z], axis=1), ones, -y, -y, grid_size)

    # yz 平面 — x 正反两个方向
    yz_pos_x = _scatter_project(np.stack([y, z], axis=1), ones, x, x, grid_size)
    yz_neg_x = _scatter_project(np.stack([y, z], axis=1), ones, -x, -x, grid_size)

    return np.stack(
        [xy_pos_z, xy_neg_z,
         xz_pos_y, xz_neg_y,
         yz_pos_x, yz_neg_x],
        axis=0)


def project(points: np.ndarray, num_views: int = 3, grid_size: int = GRID_SIZE) -> np.ndarray:
    """统一入口：根据 num_views 选择 3 视图或 6 视图投影。"""
    if num_views == 6:
        return multi_view_project_6(points, grid_size)
    return multi_view_project(points, grid_size)


# ══════════════════════════════════════════════════════════════
#  数据增强
# ══════════════════════════════════════════════════════════════

def augment_points(points: np.ndarray) -> np.ndarray:
    """点云数据增强：绕 z 轴随机旋转、缩放、平移、抖动。"""
    # 绕 z 轴随机旋转
    theta = np.random.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot_z = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1],
    ], dtype=np.float32)
    points = points @ rot_z.T

    # 随机缩放
    scale = np.random.uniform(0.9, 1.1)
    points *= scale

    # 随机平移
    shift = np.random.uniform(-0.05, 0.05, size=(1, 3))
    points += shift

    # 坐标抖动
    jitter = np.random.normal(0, 0.01, size=points.shape).astype(np.float32)
    points += jitter

    # 重新归一化
    max_abs = np.max(np.abs(points))
    if max_abs > 0:
        points /= max_abs

    return points
