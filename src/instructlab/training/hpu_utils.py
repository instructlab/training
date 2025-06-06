import os
import torch
from functools import lru_cache


@lru_cache(maxsize=None)
def is_torch_hpu_available() -> bool:
    try:
        import habana_frameworks.torch.core  # noqa: F401
    except ImportError:
        return False
    return True


def simple_bucket(length):
    """
    This bucket algorithm merely relies on the given number instead of based on
    slicing the known (min, max) range for several reasons:
        1) Due to the use of the first-fit-decreasing (FFD) algorithm, the
           (min, max) sequence length of each rank will be much smaller than the
           (min, max) sequence length of the dataset. Bucketing on the
           (min, max) sequence length of the dataset is not practical
        2) The (min, max) sequence length of a given rank is unknown until
           finishing 1 epoch since the packing is done on the fly
        3) Due to the shuffling, the (min, max) sequence length of a given rank
           may vary between ranks. Once the (min, max) sequence length of a
           given rank changes, the bucketing also needs adjustment

    This bucket algorithm is based on the most significant set bit of the input number.
    It first check what’s the most significant set bit, assuming it's bit "S",
    and then slice the range [2 ** S, 2 ** (S+1)] into buckets with the same size.
    By default the range is divided into 16 buckets, so the bucket size will be
    2 ** (S - 4)
    For example, 0b10001 will be padded to 0b10010.
    This approach can limit the overhead of bucketing (at most 1/16 of the input
    number) and also prevent recompilation due to a too small bucket size.
    """
    l = length
    msb = 0
    while l > 0:
        msb += 1
        l = l // 2

    align = (1 << (msb - 4)) if msb >= 4 else 1

    return (length + align - 1) // align * align


def bucket(length):
    return simple_bucket(length)


def save_hpu_model(model, output_dir):
    from safetensors.torch import save_file

    state_dict = model.state_dict()
    remove_prefix = "_orig_mod."
    clean_state_dict = {
        k[len(remove_prefix) :] if k.startswith(remove_prefix) else k: v
        for k, v in state_dict.items()
    }
    save_file(clean_state_dict, os.path.join(output_dir, "model.safetensors"))
