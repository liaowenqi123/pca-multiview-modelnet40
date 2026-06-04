"""
训练 / 评估工具函数。
"""

import torch


def _unpack_batch(batch, device):
    """兼容 2-tuple (views, labels) 和 3-tuple (views, points, labels)。"""
    if len(batch) == 3:
        views, points, labels = batch
        views = views.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        return (views, points), labels
    else:
        views, labels = batch
        views = views.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        return views, labels


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

    for batch in pbar:
        model_input, labels = _unpack_batch(batch, device)

        optimizer.zero_grad()
        logits = model(model_input)
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

    for batch in dataloader:
        model_input, labels = _unpack_batch(batch, device)

        logits = model(model_input)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total
