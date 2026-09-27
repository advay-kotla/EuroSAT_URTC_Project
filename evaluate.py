import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from torch.utils.data import DataLoader

from dataset import EuroSATDataset, test_transform

from models import (
    get_resnet50,
    get_efficientnet,
    get_vit
)


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

MODELS = {
    "ResNet50": (
        get_resnet50,
        "resnet50_best.pth"
    ),

    "EfficientNet-B0": (
        get_efficientnet,
        "efficientnet_best.pth"
    ),

    "Vision Transformer": (
        get_vit,
        "vit_best.pth"
    )
}


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


os.makedirs("results", exist_ok=True)

results = []


for model_name, (builder, weights) in MODELS.items():

    print(f"\nEvaluating {model_name}...")

    model = builder()

    model.load_state_dict(
        torch.load(weights, map_location=DEVICE)
    )

    model.to(DEVICE)

    model.eval()

    predictions = []
    labels_list = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            preds = outputs.argmax(dim=1).cpu().numpy()

            predictions.extend(preds)

            labels_list.extend(labels.numpy())

    accuracy = accuracy_score(
        labels_list,
        predictions
    )

    precision = precision_score(
        labels_list,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        labels_list,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        labels_list,
        predictions,
        average="weighted"
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    cm = confusion_matrix(
        labels_list,
        predictions
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    fig, ax = plt.subplots(figsize=(8,8))

    disp.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False
    )

    plt.title(model_name)

    plt.tight_layout()

    plt.savefig(
        f"results/{model_name.replace(' ','_')}_confusion_matrix.png"
    )

    plt.close()


df = pd.DataFrame(results)

df.to_csv(
    "results/model_results.csv",
    index=False
)

print("\n==============================")
print(df)
print("==============================")