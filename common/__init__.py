"""
公共工具：MIOpen/ROCm 配置、torchvision 镜像、模块导出。
"""

import os
import torch

# ─── ROCm / MIOpen 配置 ──────────────────────────────────────
def setup_rocm():
    """在 ROCm 上禁用 MIOpen JIT 编译，回退 ATen 原生实现。"""
    os.environ.setdefault("MIOPEN_FIND_MODE", "normal")
    os.environ.setdefault("MIOPEN_DEBUG_FIND_ONLY_SOLVER", "1")
    os.environ.setdefault("MIOPEN_DEBUG_GCN_ASM_KERNELS", "0")
    os.environ.setdefault("MIOPEN_DEBUG_CONV_DIRECT", "0")
    os.environ.setdefault("MIOPEN_DEBUG_CONV_WINOGRAD", "0")
    os.environ.setdefault("MIOPEN_DEBUG_CONV_FFT", "0")
    os.environ.setdefault("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM", "0")
    os.environ.setdefault("MIOPEN_ENABLE_LOGGING_CMD", "0")
    try:
        torch.backends.cudnn.enabled = False
        print("[rocm] 已禁用 cuDNN/MIOpen 后端。")
    except Exception:
        pass


# ─── torchvision 镜像 ────────────────────────────────────────
def setup_mirror():
    """Monkey-patch torch.hub 下载函数，失败时走 HF 镜像。"""
    import torch.hub as hub

    _orig_download = hub.download_url_to_file

    def _mirror_download(url, dst, hash_prefix=None, progress=True):
        try:
            return _orig_download(url, dst, hash_prefix=hash_prefix, progress=progress)
        except Exception:
            alt = url.replace(
                "https://download.pytorch.org/models/",
                "https://hf-mirror.com/pytorch/vision/resolve/main/",
            )
            if alt == url:
                raise
            print(f"[mirror] 尝试镜像: {alt}")
            try:
                return _orig_download(alt, dst, hash_prefix=hash_prefix, progress=progress)
            except Exception:
                raise

    hub.download_url_to_file = _mirror_download
    print("[mirror] torchvision 下载已配置 HF 镜像回退。")


# ─── 常用常量 ───────────────────────────────────────────────
from .preprocessing import NUM_CLASSES, NUM_POINTS, GRID_SIZE

# ─── 便捷导出 ───────────────────────────────────────────────
from .preprocessing import pca_align, multi_view_project, augment_points
from .dataset import ModelNet40MultiView
from .models import MultiViewResNet, MultiViewResNetV3, MultiViewResNetV4, MultiViewResNetV6
from .train_utils import train_one_epoch, evaluate
