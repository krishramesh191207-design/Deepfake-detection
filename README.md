# Deepfake Detection with EfficientNet-B4

Binary image classifier (real = 0, fake = 1) built by fine-tuning an ImageNet-pretrained
EfficientNet-B4 on the provided hackathon dataset only.

## Results

Validation set: 1,400 images (20% stratified split, seed 42). The checkpoint with the best
validation AUC is kept; F1 and accuracy are measured at that epoch with threshold 0.5.

| Run | Input size | Key settings | Saved epoch | Val AUC | F1 | Acc |
|---|---|---|---|---|---|---|
| v1 | 224 | baseline, mild augmentation, LR 1e-4 | 8 | 0.8749 | 0.825 | 0.824 |
| v2 | 224 | frozen warm-up, MixUp, label smoothing, blur, LR 3e-5 | 22 | 0.8574 | 0.801 | 0.794 |
| v3 | 320 | v1 recipe at higher resolution | 7 | 0.8800 | 0.834 | 0.836 |
| **final** | **380** | v1 recipe at EfficientNet-B4's native resolution | 5 | **0.8830** | **0.844** | **0.843** |

The checkpoint is also selected on this validation set, so these numbers are slightly
optimistic, and AUC differences below about 0.01 are within noise for 1,400 images.
Per-epoch logs and curves for every run are in `logs/`.

### What we learned
- Full fine-tuning of the pretrained network with AdamW (LR 1e-4) works well. Freezing the
  backbone first (v2) was weak: the head alone reached only about 0.70 val AUC.
- Heavy regularization (MixUp, label smoothing, Gaussian blur, stronger color jitter) lowered
  accuracy. Our hypothesis is that it suppresses the subtle artifacts that separate fakes from reals.
- Higher input resolution helped modestly (AUC 0.8749 -> 0.8800 -> 0.8830 for 224 -> 320 -> 380),
  even though the source images are 224x224. Upscaling adds no new information, so the gain
  comes from larger feature maps in the network, at a higher compute cost.
- The model overfits quickly: val AUC peaks around epochs 3-8 while train AUC exceeds 0.96, so
  training uses early stopping on val AUC (patience 4).
- Flip test-time augmentation changed val AUC by only +0.0001 while doubling inference cost, so
  it is off by default (`--tta` to enable).

## Repository layout

```
src/model.py       EfficientNet-B4 with a 1-logit head (Dropout 0.4)
src/dataset.py     datasets and transforms
src/train.py       training with early stopping on val AUC
src/predict.py     writes predictions.csv from a checkpoint, reports speed and memory
notebooks/         the executed Colab notebook that produced the final weights
logs/              training logs and curves for all runs
requirements.txt
```

## Setup

```
pip install -r requirements.txt
```

Expected dataset layout (not included in this repo):

```
<data_dir>/
  train_metadata.csv          columns: image_id, label
  train/real/<image_id>.jpg
  train/fake/<image_id>.jpg
  test/images/*.jpg
```

## Reproduce

Train (about 145 s per epoch at 380px on a Tesla T4; the final run stopped early at epoch 9):

```
python src/train.py --data_dir /path/to/hackathon_dataset --image_size 380 --batch_size 16 --epochs 12 --output_dir output
```

Predict on the test set:

```
python src/predict.py --data_dir /path/to/hackathon_dataset --weights output/model_weights.pt --image_size 380 --out predictions.csv
```

`--image_size` must match the value used for training. `predict.py` prints end-to-end
throughput (images/s) and peak GPU memory. The notebook's submitted CSV was produced with
flip-TTA; add `--tta` to reproduce it exactly.

The seed is fixed to 42 (Python, NumPy, PyTorch, deterministic cuDNN). Small run-to-run
differences can still occur across GPU types and library versions because of mixed-precision
training.

## Colab notebook

`notebooks/train_v3_380.ipynb` is the notebook that produced the final weights, with its
outputs saved. It contains Colab-only cells (dataset zip upload, `files.download`); skip
them when running elsewhere and set `DATA_DIR` to your dataset folder.

## Model and efficiency

- Architecture: EfficientNet-B4 (17.55M parameters), classifier replaced by Dropout(0.4) + Linear(1).
- Loss/optimizer: BCE with logits, AdamW (LR 1e-4, weight decay 1e-4), cosine schedule over 12 epochs, mixed precision.
- Input: 380x380, ImageNet normalization. Training augmentation: horizontal flip, rotation (10 degrees),
  mild color jitter, random grayscale (5%), random erasing (10%).
- Inference: one forward pass per image (no TTA by default), fp16 autocast on GPU, threshold 0.5.
  Resolution is the main cost lever: 224px needs about 2.9x fewer pixels than 380px for a small accuracy loss
  (see the results table).

## Rules compliance

Training and fine-tuning use only the provided dataset. The only external component is
torchvision's ImageNet-1K pretrained EfficientNet-B4 weights (a permitted pretrained architecture).
No external datasets were used.

## Model weights

`model_weights.pt` (PyTorch `state_dict`) is in the submission Drive folder. It is excluded from
git via `.gitignore`.
