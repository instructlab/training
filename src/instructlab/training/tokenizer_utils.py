# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, field
from typing import List

# Third Party
from transformers import AutoTokenizer

# First Party
from instructlab.training.utils import log_rank_0


from dataclasses import dataclass, field

@dataclass
class TokenInfo:
    token: str
    add_to_tokenizer: bool = False

@dataclass
class SpecialTokens:
    system: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    user: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    assistant: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    eos: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    pad: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    bos: TokenInfo = field(default_factory=lambda: TokenInfo(""))

    def get_tokens_to_add(self) -> List[str]:
        return [
            token_info.token
            for token_info in self.__dict__.values()
            if token_info.add_to_tokenizer and token_info.token
        ]


def setup_tokenizer(model_name_or_path, SPECIAL_TOKENS, CHAT_TEMPLATE):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)

    if not SPECIAL_TOKENS.pad:
        SPECIAL_TOKENS.pad = SPECIAL_TOKENS.eos
    tokenizer.add_special_tokens(
        {
            "bos_token": SPECIAL_TOKENS.bos.token,
            "eos_token": SPECIAL_TOKENS.eos.token,
            "pad_token": SPECIAL_TOKENS.pad.token,
        }
    )
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS.get_tokens_to_add()})
    if getattr(tokenizer, "add_bos_token", False) or getattr(
        tokenizer, "add_eos_token", False
    ):
        log_rank_0(
            "\033[91m!!!!!!!! tokenizer has add_bos_token or add_eos_token\033[0m"
        )
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    tokenizer.chat_template = CHAT_TEMPLATE
    assert len(get_sp_token(tokenizer, SPECIAL_TOKENS.eos.token)) == 1
    assert len(get_sp_token(tokenizer, SPECIAL_TOKENS.pad.token)) == 1
    return tokenizer


def get_sp_token(tokenizer, sp_string):
    sp_token = tokenizer.encode(sp_string, add_special_tokens=False)
    # assert 1 == len(sp_token)
    return sp_token
