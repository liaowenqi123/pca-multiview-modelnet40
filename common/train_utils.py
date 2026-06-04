"""
训练 / 评估工具函数。
"""

import torch


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个 epoch，返回 (avg_loss, accuracy)。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d} [train]", leave=False)
    except ImportError:
        pbar = dataloader

    for batch_idx, (views, labels) in enumerate(pbar):
        views = views.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(views)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        if hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "acc": f"{correct/total:.3f}"})

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """评估，返回 (avg_loss, accuracy)。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for views, labels in dataloader:
        views = views.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(views)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total
