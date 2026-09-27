import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from dataset import EuroSATDataset, train_transform, test_transform

from models import (
    get_resnet50,
    get_efficientnet,
    get_vit
)


# ==========================
# SETTINGS
# ==========================

MODEL_NAME = "vit"

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", DEVICE)


# ==========================
# DATA
# ==========================

train_dataset = EuroSATDataset(
    csv_file="EuroSAT/train.csv",
    root_dir="EuroSAT",
    transform=train_transform
)


val_dataset = EuroSATDataset(
    csv_file="EuroSAT/validation.csv",
    root_dir="EuroSAT",
    transform=test_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================
# MODEL
# ==========================

if MODEL_NAME == "resnet50":
    model = get_resnet50()

elif MODEL_NAME == "efficientnet":
    model = get_efficientnet()

elif MODEL_NAME == "vit":
    model = get_vit()

else:
    raise ValueError("Invalid model")


model = model.to(DEVICE)


# ==========================
# TRAINING
# ==========================

criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


best_accuracy = 0

train_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)


        optimizer.zero_grad()


        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()



    train_accuracy = correct / total



    # Validation

    model.eval()

    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

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



    val_accuracy = correct / total


    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )

    print(
        f"Loss: {total_loss/len(train_loader):.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )
    train_losses.append(total_loss/len(train_loader))
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    # Save best model

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            f"{MODEL_NAME}_best.pth"
        )


print("Training complete!")
print("Best validation accuracy:", best_accuracy)
plt.figure(figsize=(8,5))

plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

plt.savefig("training_loss_curve.png")
plt.close()


plt.figure(figsize=(8,5))

plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy Curve")
plt.legend()

plt.savefig("accuracy_curve.png")
plt.close()


print("Training curves saved!")