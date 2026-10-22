"""One call that fixes every source of randomness, so each reported number can be reproduced."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic cuDNN kernels: slightly slower, but the same seed gives the same model.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
