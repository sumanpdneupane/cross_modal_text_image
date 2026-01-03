import torch

def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        print("Using GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple MPS")
    else:
        device = "cpu"
        print("Using CPU")
    return device

def clean_memory(device="cpu"):
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def chunk_text(text, chunk_size=128):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())