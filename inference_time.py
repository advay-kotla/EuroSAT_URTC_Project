import torch
import time

from torch.utils.data import DataLoader

from dataset import EuroSATDataset, test_transform

from models import (
    get_resnet50,
    get_efficientnet,
    get_vit
)


DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using device:", DEVICE)


# Test dataset
test_dataset = EuroSATDataset(
    csv_file="EuroSAT/test.csv",
    root_dir="EuroSAT",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


models = {
    "ResNet50": (get_resnet50(), "resnet50_best.pth"),
    "EfficientNet-B0": (get_efficientnet(), "efficientnet_best.pth"),
    "Vision Transformer": (get_vit(), "vit_best.pth")
}


for name, (model, weights) in models.items():

    print("\n===================")
    print(name)

    model.load_state_dict(
        torch.load(weights, map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()


    total_time = 0
    count = 0


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)


            if DEVICE == "mps":
                torch.mps.synchronize()


            start = time.time()


            outputs = model(images)


            if DEVICE == "mps":
                torch.mps.synchronize()


            end = time.time()


            total_time += (end - start)
            count += 1


            # only need enough samples
            if count == 500:
                break


    avg_time = total_time / count

    print(
        f"Average inference time: {avg_time*1000:.4f} ms/image"
    )

    print(
        f"Images per second: {1/avg_time:.2f}"
    )