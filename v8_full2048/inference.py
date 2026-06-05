"""
V8 推理 — 复用 V7 模型，n_sample=2048 无采样。
"""
import sys, json, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import setup_rocm; setup_rocm()

import numpy as np, torch
from common.preprocessing import pca_align, project, GRID_SIZE, IMAGENET_MEAN, IMAGENET_STD
from common import MultiViewResNetV7, NUM_CLASSES

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "v8" / "best_model.pth"


def main():
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pts = np.load(ROOT / "test_points.npy"); N = len(pts)
    print(f"[V8] {N} samples on {d}")

    model = MultiViewResNetV7(backbone="resnet18", num_views=6, symmetric_fusion=True,
                              pretrained=False, dropout=0.5, n_sample=2048)
    model.load_state_dict(torch.load(CKPT, map_location=d)["model_state_dict"])
    model = model.to(d).eval()

    bs = 32; preds = []
    for s in range(0, N, bs):
        e = min(s+bs, N)
        v_list, p_list = [], []
        for i in range(s, e):
            p = pca_align(pts[i])
            v = project(p, num_views=6, grid_size=GRID_SIZE)
            v_list.append(torch.from_numpy((v - IMAGENET_MEAN) / IMAGENET_STD))
            p_list.append(torch.from_numpy(p))
        vb = torch.stack(v_list).to(d); pb = torch.stack(p_list).to(d)
        preds.extend(model((vb, pb)).argmax(1).cpu().tolist())
        print(f"  {e}/{N}" if e % 500 == 0 or e == N else "", end="")

    r = {str(i): int(p) for i, p in enumerate(preds)}
    jp = ROOT / "result.json"
    json.dump(r, open(jp, "w"))
    with zipfile.ZipFile(ROOT / "result.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(jp, "result.json")
    print(f"\nDone: {ROOT/'result.zip'}")


if __name__ == "__main__":
    main()
