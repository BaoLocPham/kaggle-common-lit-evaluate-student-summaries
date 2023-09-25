import random
import numpy as np
import torch
from sklearn.utils import check_random_state

def set_random_seed(seed):
    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for numpy
    np.random.seed(seed)

    # Set seed for scikit-learn
    random_state = check_random_state(seed)

    # Set seed for PyTorch on both CPU and CUDA (if available)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_everything(seed: int):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

def get_logger(filename='training_stage_1'):
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger