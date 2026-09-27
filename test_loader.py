from dataset import EuroSATDataset, train_transform


dataset = EuroSATDataset(
    csv_file="EuroSAT/train.csv",
    root_dir="EuroSAT",
    transform=train_transform
)


image, label = dataset[0]

print("Image shape:", image.shape)
print("Label:", label)