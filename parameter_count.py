import torch

from models import (
    get_resnet50,
    get_efficientnet,
    get_vit
)

models = {
    "ResNet50": get_resnet50(),
    "EfficientNet-B0": get_efficientnet(),
    "Vision Transformer": get_vit()
}


for name, model in models.items():

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n", name)
    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)