import torch.nn as nn
import torchvision.models as models


NUM_CLASSES = 10


def get_resnet50():

    model = models.resnet50(weights="IMAGENET1K_V2")

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    return model



def get_efficientnet():

    model = models.efficientnet_b0(
        weights="IMAGENET1K_V1"
    )

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    return model



def get_vit():

    model = models.vit_b_16(
        weights="IMAGENET1K_V1"
    )

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.heads.head = nn.Linear(
        model.heads.head.in_features,
        NUM_CLASSES
    )

    return model