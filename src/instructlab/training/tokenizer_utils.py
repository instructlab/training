# SPDX-License-Identifier: Apache-2.0

# Third Party
from transformers import AutoTokenizer, PreTrainedTokenizer

# First Party
from instructlab.training.utils import log_rank_0


def setup_tokenizer(
    model_name_or_path, SPECIAL_TOKENS, CHAT_TEMPLATE
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)

    if not SPECIAL_TOKENS.pad.token:
        SPECIAL_TOKENS.pad = SPECIAL_TOKENS.eos
    tokenizer.add_special_tokens(
        {
            "bos_token": SPECIAL_TOKENS.bos.token,
            "eos_token": SPECIAL_TOKENS.eos.token,
            "pad_token": SPECIAL_TOKENS.pad.token,
        }
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS.get_tokens_to_add()}
    )
    if getattr(tokenizer, "add_bos_token", False) or getattr(
        tokenizer, "add_eos_token", False
    ):
        log_rank_0(
            "\033[91m!!!!!!!! tokenizer has add_bos_token or add_eos_token\033[0m"
        )
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    tokenizer.chat_template = CHAT_TEMPLATE
    assert (
        len(get_sp_token(tokenizer, SPECIAL_TOKENS.eos.token)) == 1
    ), "EOS token doesn't exist or is of incorrect length"
    assert (
        len(get_sp_token(tokenizer, SPECIAL_TOKENS.pad.token)) == 1
    ), "Padding token doesn't exist or is of incorrect length"
    return tokenizer


def get_sp_token(tokenizer, sp_string):
    sp_token = tokenizer.encode(sp_string, add_special_tokens=False)
    # assert 1 == len(sp_token)
    return sp_token
