"""
V4: TinyCNN + 十二面体 6 视图推理脚本。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import setup_rocm
setup_rocm()

import json
import zipfile

import numpy as np
import torch

from common.preprocessing import pca_align, multi_view_project_dodeca, NORM_5CH_MEAN, NORM_5CH_STD
from common import MultiViewResNetV4, NUM_CLASSES

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "v4" / "best_model.pth"
GRID_SIZE = 56


def preprocess(points):
    aligned = pca_align(points)
    views = multi_view_project_dodeca(aligned, GRID_SIZE)
    views = (views - NORM_5CH_MEAN) / NORM_5CH_STD
    return torch.from_numpy(views)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    test_points = np.load(ROOT_DIR / "test_points.npy")
    N = len(test_points)
    print(f"[数据] 测试样本: {N}")

    print(f"[模型] 加载: {CHECKPOINT_PATH}")
    model = MultiViewResNetV4(in_channels=5, base_ch=32, dim_reduce=32, dropout=0.5)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[模型] best_acc={ckpt.get('best_acc', 'N/A')}, epoch={ckpt.get('epoch', 'N/A')}")

    batch_size = 128
    all_preds = []
    print(f"[推理] batch_size={batch_size} ...")

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        views_list = [preprocess(test_points[i]) for i in range(start, end)]
        batch = torch.stack(views_list, dim=0).to(device)

        with torch.no_grad():
            preds = model(batch).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())

        if end % 500 == 0 or end == N:
            print(f"  {end}/{N}")

    result = {str(i): int(p) for i, p in enumerate(all_preds)}
    json_path = ROOT_DIR / "result.json"
    with open(json_path, "w") as f:
        json.dump(result, f)
    print(f"[保存] {json_path}")

    zip_path = ROOT_DIR / "result.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="result.json")
    print(f"[打包] {zip_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
