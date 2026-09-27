import torch
from torch.utils.data import DataLoader

from dataset import EuroSATDataset, test_transform

from models import (
    get_resnet50,
    get_efficientnet,
    get_vit
)


# ==========================
# SETTINGS
# ==========================

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", DEVICE)


# ==========================
# TEST DATA
# ==========================

test_dataset = EuroSATDataset(
    csv_file="EuroSAT/test.csv",
    root_dir="EuroSAT",
    transform=test_transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# ==========================
# FUNCTION
# ==========================

def evaluate(model, model_name, weights):

    print("\n===================")
    print(model_name)

    model.load_state_dict(
        torch.load(
            weights,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)


            outputs = model(images)


            _, predicted = torch.max(
                outputs,
                1
            )


            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()


    accuracy = correct / total


    print(
        f"Test Accuracy: {accuracy:.4f}"
    )

    print(
        f"Correct: {correct}/{total}"
    )


# ==========================
# RUN MODELS
# ==========================


evaluate(
    get_resnet50(),
    "ResNet50",
    "resnet50_best.pth"
)


evaluate(
    get_efficientnet(),
    "EfficientNet-B0",
    "efficientnet_best.pth"
)


evaluate(
    get_vit(),
    "Vision Transformer",
    "vit_best.pth"
)