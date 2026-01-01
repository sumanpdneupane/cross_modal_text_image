from PIL import Image
import matplotlib.pyplot as plt


def show_image(img_path):
    img = Image.open(img_path).convert("RGB")
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis('off')  # hide axes
    plt.show()


def plot_training_n_validation_loss(epoch, training_loss, validation_loss, save_path=None):
    plt.figure(figsize=(5, 3))
    plt.plot(epoch, training_loss, label="Training Loss", marker='o', color='blue')
    plt.plot(epoch, validation_loss, label="Validation Loss", marker='s', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_retrieval_metrics(metrics_data, metric_name="R@1", save_path=None):
    plt.figure(figsize=(8, 5))

    for key, data in metrics_data.items():
        plt.plot(data["epoch"], data[metric_name], marker='o', label=key)

    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Across Epochs")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
