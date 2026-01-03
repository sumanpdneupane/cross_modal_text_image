import csv
import os
import torch
from src.model import FusionModel

# Save and Load Model
def save_model(model: FusionModel, optimizer, epoch, path="fusion_model.pth"):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"Model saved to {path}")


def load_model(model: FusionModel, optimizer, path="fusion_model.pth", device="cpu"):
    if not os.path.exists(path):
        return model, optimizer, 1

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    last_epoch = checkpoint.get("epoch", 1)

    print(f"Model loaded from {path} (last epoch: {last_epoch})")
    return model, optimizer, last_epoch


# Save and Load Train/Validate Loss
def save_training_n_validation_loss(epoch, train_loss, val_loss, folder="metrics",
                                    filename="training_and_validation_loss_log.csv"):
    os.makedirs(folder, exist_ok=True)
    summary_csv = os.path.join(folder, filename)
    summary_exists = os.path.exists(summary_csv)

    with open(summary_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow(["Epoch", "Training Loss", "Validation Loss"])
        writer.writerow([epoch, train_loss, val_loss])
    print(f"Loss saved to {summary_csv}")


def load_training_n_validation_loss(folder="metrics", filename="training_and_validation_loss_log.csv"):
    summary_csv = os.path.join(folder, filename)
    epochs, training_losses, validation_losses = [], [], []

    if not os.path.exists(summary_csv):
        print(f"No metrics summary found at {summary_csv}")
        return epochs, training_losses, validation_losses

    with open(summary_csv, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["Epoch"]))
            training_losses.append(float(row["Training Loss"]))
            validation_losses.append(float(row["Validation Loss"]))
    print(f"Loss loaded from {summary_csv}")
    return epochs, training_losses, validation_losses

# Save and Load Retrieval Metrics
def save_retrieval_metrics(epoch, val_results, folder="metrics", filename="retrieval_metrics_log.csv"):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    fieldnames = ["Epoch", "Type", "R@1", "R@5", "R@10"]

    # If file doesn't exist, create and write header
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    # Append new metrics
    with open(file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for key, metrics in val_results.items():
            if key != "val_loss":
                writer.writerow({
                    "Epoch": epoch,
                    "Type": key,
                    "R@1": metrics['R@1'],
                    "R@5": metrics['R@5'],
                    "R@10": metrics['R@10']
                })
    print(f"Retrieval Metrics saved to {file_path}")

def load_retrieval_metrics(folder="metrics", filename="retrieval_metrics_log.csv"):
    import csv
    import os

    file_path = os.path.join(folder, filename)
    if not os.path.exists(file_path):
        print(f"No retrieval metrics found at {file_path}")
        return {}

    metrics_data = {}
    with open(file_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row["Epoch"])
            metric_type = row["Type"]
            if metric_type not in metrics_data:
                metrics_data[metric_type] = {"epoch": [], "R@1": [], "R@5": [], "R@10": []}
            metrics_data[metric_type]["epoch"].append(epoch)
            metrics_data[metric_type]["R@1"].append(float(row["R@1"]))
            metrics_data[metric_type]["R@5"].append(float(row["R@5"]))
            metrics_data[metric_type]["R@10"].append(float(row["R@10"]))
    print(f"Retrieval Metrics loaded from {file_path}")
    return metrics_data