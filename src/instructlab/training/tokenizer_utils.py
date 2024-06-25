# Standard
from dataclasses import dataclass, field

# Third Party
from transformers import AutoTokenizer

# First Party
from instructlab.training.utils import log_rank_0


@dataclass
class SpecialTokens:
    system: str = field(default="")
    user: str = field(default="<|user|>")
    assistant: str = field(default="<|assistant|>")
    eos: str = field(default="<|endoftext|>")
    pad: str = field(default="")
    bos: str = field(default="<|begginingoftext|>")


def setup_tokenizer(model_name_or_path, SPECIAL_TOKENS, CHAT_TEMPLATE):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)

    if not SPECIAL_TOKENS.pad:
        SPECIAL_TOKENS.pad = SPECIAL_TOKENS.eos
    tokenizer.add_special_tokens(
        {
            "bos_token": SPECIAL_TOKENS.bos,
            "eos_token": SPECIAL_TOKENS.eos,
            "pad_token": SPECIAL_TOKENS.pad,
        }
    )

    if SPECIAL_TOKENS.system:
        add_token_list = [SPECIAL_TOKENS.system]
    else:
        add_token_list = []
    add_token_list.extend([SPECIAL_TOKENS.user, SPECIAL_TOKENS.assistant])

    tokenizer.add_special_tokens({"additional_special_tokens": add_token_list})
    if getattr(tokenizer, "add_bos_token", False) or getattr(
        tokenizer, "add_eos_token", False
    ):
        log_rank_0(
            "\033[91m!!!!!!!! tokenizer has add_bos_token or add_eos_token\033[0m"
        )
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def get_sp_token(tokenizer, sp_string):
    sp_token = tokenizer.encode(sp_string, add_special_tokens=False)
    assert 1 == len(sp_token)
    return sp_token[0]
