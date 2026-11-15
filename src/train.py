"""Fine-tune EfficientNet-B4 for binary deepfake detection (real = 0, fake = 1).

Example:
    python src/train.py --data_dir /path/to/hackathon_dataset --image_size 380 \
        --batch_size 16 --epochs 12 --output_dir output

The checkpoint with the best validation AUC is saved to <output_dir>/model_weights.pt.
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import DeepfakeDataset, get_transforms
from model import build_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', required=True,
                   help='folder containing train_metadata.csv, train/{real,fake}/ and test/images/')
    p.add_argument('--output_dir', default='output')
    p.add_argument('--image_size', type=int, default=380, help='square input size; must match at inference')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=4, help='early-stopping patience (epochs) on val AUC')
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--threshold', type=float, default=0.5, help='decision threshold for F1/accuracy')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--no_pretrained', action='store_true',
                   help='random init instead of ImageNet weights (for offline smoke tests only)')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    total_loss, all_labels, all_probs = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        all_probs.extend(torch.sigmoid(logits).detach().float().cpu().numpy().flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())
    return total_loss / len(loader.dataset), roc_auc_score(all_labels, all_probs)


@torch.no_grad()
def validate_epoch(model, loader, criterion, device, use_amp, threshold):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
        with torch.autocast(device_type=device, enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        all_probs.extend(torch.sigmoid(logits).float().cpu().numpy().flatten().tolist())
        all_labels.extend(labels.cpu().numpy().flatten().tolist())
    auc = roc_auc_score(all_labels, all_probs)
    preds = (np.array(all_probs) >= threshold).astype(int)
    return (total_loss / len(loader.dataset), auc,
            f1_score(all_labels, preds), accuracy_score(all_labels, preds))


def save_curves(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    log = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(log['epoch'], log['tr_loss'], label='Train Loss')
    axes[0].plot(log['epoch'], log['va_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Loss curves'); axes[0].legend()
    axes[1].plot(log['epoch'], log['tr_auc'], label='Train AUC')
    axes[1].plot(log['epoch'], log['va_auc'], label='Val AUC')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('AUC-ROC')
    axes[1].set_title('AUC-ROC curves'); axes[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Device: {device} | image size: {args.image_size} | batch: {args.batch_size} | output: {out_dir}')

    metadata = pd.read_csv(data_dir / 'train_metadata.csv')
    print('Total samples:', len(metadata))
    print(metadata['label'].value_counts())
    train_df, val_df = train_test_split(
        metadata, test_size=args.val_split, stratify=metadata['label'], random_state=args.seed)
    print(f'Train: {len(train_df)} | Val: {len(val_df)}')

    train_tf, eval_tf = get_transforms(args.image_size)
    train_loader = DataLoader(DeepfakeDataset(train_df, data_dir, train_tf), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=use_amp)
    val_loader = DataLoader(DeepfakeDataset(val_df, data_dir, eval_tf), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=use_amp)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = make_scaler(use_amp)

    best_auc, no_improve, history = 0.0, 0, []
    best_path = out_dir / 'model_weights.pt'

    cols = ['Epoch', 'TrLoss', 'TrAUC', 'VaLoss', 'VaAUC', 'F1', 'Acc']
    print(''.join(f'{c:>9}' for c in cols))
    print('-' * 70)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc = train_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        va_loss, va_auc, va_f1, va_acc = validate_epoch(
            model, val_loader, criterion, device, use_amp, args.threshold)
        scheduler.step()

        vals = [tr_loss, tr_auc, va_loss, va_auc, va_f1, va_acc]
        print(f'{epoch:>9}' + ''.join(f'{v:>9.4f}' for v in vals) + f'   ({time.time() - t0:.0f}s)')
        history.append({'epoch': epoch, 'tr_loss': tr_loss, 'tr_auc': tr_auc, 'va_loss': va_loss,
                        'va_auc': va_auc, 'va_f1': va_f1, 'va_acc': va_acc})

        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(model.state_dict(), best_path)
            print(f'  Best AUC: {best_auc:.4f} - saved')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

    pd.DataFrame(history).to_csv(out_dir / 'training_log.csv', index=False)
    save_curves(history, out_dir / 'training_curves.png')
    print(f'\nBest val AUC: {best_auc:.4f}  (weights: {best_path})')


if __name__ == '__main__':
    main()
