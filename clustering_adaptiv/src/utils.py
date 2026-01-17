import random
import numpy as np
import torch

MEAN = (0.6556356704765691, 0.5643559072204862, 0.5039317917282723)
STD = (0.3247947359540185, 0.3724843070555076, 0.40141975985491773)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_image(arr):
    mean = np.array(MEAN, dtype=np.float32)
    std = np.array(STD, dtype=np.float32)
    return (arr - mean) / std


def to_tensor(arr):
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def split_train_val(n, val_ratio, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_size = int(n * val_ratio)
    return indices[val_size:], indices[:val_size]
