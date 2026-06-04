"""
V6: V3 + PointNet 交叉注意力推理脚本。
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

from common.preprocessing import pca_align, project, GRID_SIZE, IMAGENET_MEAN, IMAGENET_STD
from common import MultiViewResNetV6, NUM_CLASSES

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "v6" / "best_model.pth"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    test_points_raw = np.load(ROOT_DIR / "test_points.npy")
    N = len(test_points_raw)
    print(f"[数据] 测试样本: {N}")

    print(f"[模型] 加载: {CHECKPOINT_PATH}")
    model = MultiViewResNetV6(
        backbone="resnet18", num_views=6, symmetric_fusion=True,
        pretrained=False, dropout=0.5,
    )
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[模型] best_acc={ckpt.get('best_acc', 'N/A')}, epoch={ckpt.get('epoch', 'N/A')}")

    batch_size = 32
    all_preds = []
    print(f"[推理] batch_size={batch_size} ...")

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        views_list = []
        points_list = []
        for i in range(start, end):
            pts = pca_align(test_points_raw[i])               # (2048, 3)
            views = project(pts, num_views=6, grid_size=GRID_SIZE)  # (6, 3, 224, 224)
            views = (views - IMAGENET_MEAN) / IMAGENET_STD
            views_list.append(torch.from_numpy(views))
            points_list.append(torch.from_numpy(pts))

        views_batch = torch.stack(views_list, dim=0).to(device)
        points_batch = torch.stack(points_list, dim=0).to(device)

        with torch.no_grad():
            preds = model((views_batch, points_batch)).argmax(dim=1).cpu().numpy()
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
