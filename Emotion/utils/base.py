import ast
import random

import numpy as np

import torch
from transformers.file_utils import is_torch_available


def set_seed(seed: int):
    
    """
    Helper function for reproducible behavior to set the seed in "random", "numpy", "torch".

    Args:
        seed (int): 
            The seed to set.
    """

    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def str_to_python(input_content_str: str):
    """ 
    Converts pythonic content surrounded by string literal into just content.
    '[a, b, c]' -> [a, b, c]
    """
    return ast.literal_eval(input_content_str)
