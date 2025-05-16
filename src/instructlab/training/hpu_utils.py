import torch

from functools import lru_cache
@lru_cache(maxsize=None)
def is_torch_hpu_available() -> bool:
    try:
        import habana_frameworks.torch.core  # noqa: F401
    except ImportError:
        return False
    return True
    
