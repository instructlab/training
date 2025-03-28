# SPDX-License-Identifier: Apache-2.0

# Third Party
from transformers import AutoTokenizer, PreTrainedTokenizer

# First Party
from instructlab.training.utils import log_rank_0, retrieve_chat_template


def setup_tokenizer_with_existing_chat_template(
    tokenizer: PreTrainedTokenizer,
) -> PreTrainedTokenizer:
    # otherwise, when the user doesn't provide a chat template path, we will use the default chat template
    assert (
        tokenizer.eos_token is not None
    ), "provided chat template doesn't have an EOS token, need to handle this case"
    if not tokenizer.pad_token:
        # we need to set the padding token
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # ensure the pad token is in the additional special tokens without duplicating anything else
    new_tokens = []
    if tokenizer.pad_token not in tokenizer.additional_special_tokens:
        new_tokens.append(tokenizer.pad_token)
    if tokenizer.eos_token not in tokenizer.additional_special_tokens:
        new_tokens.append(tokenizer.eos_token)

    # ensure the tokens are being sorted to prevent any issues
    new_tokens = sorted(new_tokens)
    additional_special_tokens = tokenizer.additional_special_tokens + new_tokens
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )

    # ensure the necessary tokens exist
    assert (
        len(get_sp_token(tokenizer, tokenizer.pad_token)) == 1
    ), "padding token doesn't exist or is of incorrect length"
    assert (
        len(get_sp_token(tokenizer, tokenizer.eos_token)) == 1
    ), "EOS token doesn't exist or is of incorrect length"
    return tokenizer


def setup_tokenizer_from_new_chat_template(
    tokenizer: PreTrainedTokenizer,
    chat_tmpl_path: str,
) -> PreTrainedTokenizer:
    CHAT_TEMPLATE, SPECIAL_TOKENS = retrieve_chat_template(chat_tmpl_path)
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


def setup_tokenizer(
    model_name_or_path,
    chat_tmpl_path: str | None = None,
) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)
    if not tokenizer.chat_template and chat_tmpl_path is None:
        raise ValueError(
            "Tokenizer does not have a chat template. Please provide a path to a chat template."
        )
    if not chat_tmpl_path:
        return setup_tokenizer_with_existing_chat_template(tokenizer)
    return setup_tokenizer_from_new_chat_template(tokenizer, chat_tmpl_path)


def get_sp_token(tokenizer, sp_string):
    sp_token = tokenizer.encode(sp_string, add_special_tokens=False)
    # assert 1 == len(sp_token)
    return sp_token
