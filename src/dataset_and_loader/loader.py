import random
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset

def get_transform(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return train_transform, val_transform


def dataset_loader(full_dataset, train_transform, val_transform, batch_size, seed=42):
    train_ratio = 0.4

    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    all_indices = list(range(dataset_size))

    # Shuffle indices for unbiased split but reproducible
    random.seed(seed)
    # random.shuffle(all_indices)

    # Split
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    # Subsets
    train_dataset = Subset(full_dataset, train_indices)
    # val_dataset = Subset(full_dataset, val_indices)
    val_dataset = Subset(full_dataset, train_indices)

    # Set transforms for subsets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    return train_loader, val_loader