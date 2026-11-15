"""Model definition: ImageNet-pretrained EfficientNet-B4 with a 1-logit head."""
import torch.nn as nn
from torchvision.models import EfficientNet_B4_Weights, efficientnet_b4


def build_model(pretrained=True, dropout=0.4):
    """Return EfficientNet-B4 whose classifier outputs a single logit (fake vs. real).

    pretrained=True  -> start from torchvision's ImageNet-1K weights (used for training).
    pretrained=False -> random init; use this when loading our own checkpoint at
                        inference time (no download needed, the checkpoint overwrites it).
    """
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b4(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    return model
