import torch


def get_device() -> str:
    """CUDA (Kaggle/Colab) -> MPS (Apple GPU, your M5) -> CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
