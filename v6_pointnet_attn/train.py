"""
V6: V3 + PointNet 交叉注意力路径。

    V3 路径:  6 views → ResNet18 → symmetric fusion → 门控 + 融合分类
    PointNet: 原始点云 → 自注意力(无embedding) → PointNet → K, V
              V3 的 3 个融合视角特征 → Q
              cross-attention(Q, K, V) → 投影 → logits_pn
    最终:     logits_v3 + logits_pn
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import setup_rocm, setup_mirror
setup_rocm()
setup_mirror()

import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from common import (
    NUM_CLASSES, GRID_SIZE,
    ModelNet40MultiView,
    MultiViewResNetV6,
    train_one_epoch,
    evaluate,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "v6"


def main():
    parser = argparse.ArgumentParser(description="V6: V3 + PointNet cross-attention")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dim_reduce", type=int, default=256, help="V3 融合投影维度")
    parser.add_argument("--d_attn", type=int, default=128, help="点云自注意力维度")
    parser.add_argument("--d_pointnet", type=int, default=512, help="PointNet 特征维")
    parser.add_argument("--d_kv", type=int, default=128, help="交叉注意力 K/V 维")
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--no_cache_pca", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"[设备] {device}")

    # ── 数据 ──
    train_points = np.load(ROOT_DIR / "train_points.npy")
    train_labels = np.load(ROOT_DIR / "train_labels.npy")
    test_points = np.load(ROOT_DIR / "test_points.npy")

    test_labels_path = ROOT_DIR / "test_labels.npy"
    if test_labels_path.exists():
        test_labels = np.load(test_labels_path)
    else:
        np.random.seed(42)
        indices = np.random.permutation(len(train_points))
        split = int(len(train_points) * 0.8)
        train_idx, val_idx = indices[:split], indices[split:]
        test_points = train_points[val_idx]
        test_labels = train_labels[val_idx]
        train_points = train_points[train_idx]
        train_labels = train_labels[train_idx]

    print(f"[数据] 训练: {len(train_points)}, 验证: {len(test_points)}")

    # ── 数据集 (V6 需要 return_aligned) ──
    train_dataset = ModelNet40MultiView(
        train_points, train_labels, num_views=6, projection="ortho",
        grid_size=GRID_SIZE, augment=True,
        cache_pca=not args.no_cache_pca, return_aligned=True,
    )
    test_dataset = ModelNet40MultiView(
        test_points, test_labels, num_views=6, projection="ortho",
        grid_size=GRID_SIZE, augment=False,
        cache_pca=not args.no_cache_pca, return_aligned=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )

    # ── 模型 ──
    model = MultiViewResNetV6(
        backbone="resnet18", num_views=6, symmetric_fusion=True,
        dim_reduce=args.dim_reduce,
        d_attn=args.d_attn, d_pointnet=args.d_pointnet, d_kv=args.d_kv,
        pretrained=not args.no_pretrain, dropout=args.dropout,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 参数: {total:,} (可训练 {trainable:,})")
    print(f"[模型] V3 融合输入: {model.num_fused * args.dim_reduce}d")
    print(f"[模型] PointNet attn dims: d_attn={args.d_attn} d_pn={args.d_pointnet} d_kv={args.d_kv}")

    # ── 优化器 ──
    optimizer = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.view_projections.parameters(), "lr": args.lr * 0.1},
        {"params": model.classifier.parameters(), "lr": args.lr},
        {"params": model.confidence_proj.parameters(), "lr": args.lr},
        {"params": model.view_classifiers.parameters(), "lr": args.lr},
        {"params": model.point_path.parameters(), "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", optimizer.state_dict()))
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"[恢复] epoch {start_epoch}, best_acc {best_acc:.4f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"V6 V3+PointNet Cross-Attn | {args.epochs} eps | bs={args.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, CHECKPOINT_DIR / "best_model.pth")
            print(f"  >>> 保存最佳模型 (acc={best_acc:.4f})")

        if (epoch + 1) % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc,
            }, CHECKPOINT_DIR / f"epoch_{epoch+1}.pth")

    print(f"\n{'='*60}")
    print(f"训练结束。最佳验证准确率: {best_acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
