"""Generate predictions.csv (image_id, label, confidence) with a trained checkpoint.

Example:
    python src/predict.py --data_dir /path/to/hackathon_dataset \
        --weights output/model_weights.pt --image_size 380 --out predictions.csv

--image_size MUST match the value used for training. Pass --tta to also average with a
horizontally flipped view (reproduces the notebook's CSV; ~2x slower, +0.0001 val AUC).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, get_transforms
from model import build_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', help='dataset root; test images are read from <data_dir>/test/images')
    p.add_argument('--image_dir', help='folder of test images (overrides --data_dir)')
    p.add_argument('--weights', required=True, help='path to model_weights.pt')
    p.add_argument('--image_size', type=int, default=380)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--tta', action='store_true', help='average with a horizontal-flip view')
    p.add_argument('--out', default='predictions.csv')
    p.add_argument('--num_workers', type=int, default=2)
    return p.parse_args()


@torch.no_grad()
def predict(model, loader, device, use_amp, tta):
    probs_all, ids_all = [], []
    for imgs, ids in loader:
        imgs = imgs.to(device)
        with torch.autocast(device_type=device, enabled=use_amp):
            logits = model(imgs)
            if tta:
                logits = (logits.float() + model(torch.flip(imgs, dims=[3])).float()) / 2
        probs_all.extend(torch.sigmoid(logits.float()).squeeze(-1).cpu().numpy().tolist())
        ids_all.extend(list(ids))
    return np.array(probs_all), ids_all


def main():
    args = parse_args()
    if args.image_dir:
        image_dir = Path(args.image_dir)
    elif args.data_dir:
        image_dir = Path(args.data_dir) / 'test' / 'images'
    else:
        raise SystemExit('Provide --data_dir or --image_dir')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'

    model = build_model(pretrained=False)
    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    _, eval_tf = get_transforms(args.image_size)
    dataset = TestDataset(image_dir, eval_tf)
    if len(dataset) == 0:
        raise SystemExit(f'No .jpg/.png images found in {image_dir}')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=use_amp)
    print(f'Device: {device} | images: {len(dataset)} | image size: {args.image_size} | TTA: {args.tta}')

    if use_amp:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    probs, ids = predict(model, loader, device, use_amp, args.tta)
    if use_amp:
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    pred_df = pd.DataFrame({
        'image_id': ids,
        'label': (probs >= args.threshold).astype(int).tolist(),
        'confidence': probs.tolist(),
    })
    pred_df.to_csv(args.out, index=False)

    print(f'Saved {args.out}')
    print(f'FAKE: {int(pred_df["label"].sum())}  REAL: {int((pred_df["label"] == 0).sum())}  '
          f'mean confidence: {pred_df["confidence"].mean():.4f}')
    print(f'Inference (end-to-end, incl. image loading): {elapsed:.1f}s '
          f'= {len(dataset) / elapsed:.1f} images/s')
    if use_amp:
        print(f'Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB')


if __name__ == '__main__':
    main()
