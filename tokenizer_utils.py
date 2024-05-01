# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass, field

# Third Party
from transformers import AutoTokenizer

# First Party
from utils import log_rank_0


@dataclass
class SpecialTokens:
    system: str = field(default="<|system|>")
    user: str = field(default="<|user|>")
    assistant: str = field(default="<|assistant|>")
    eos: str = field(default="<|endoftext|>")
    pad: str = field(default="<|pad|>")


SPECIAL_TOKENS = SpecialTokens()

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{'<|system|>'+ '\n' + message['content'] + '\n'}}"
    "{% elif message['role'] == 'user' %}"
    "{{'<|user|>' + '\n' + message['content'] + '\n'}}"
    "{% elif message['role'] == 'assistant' %}"
    "{{'<|assistant|>' + '\n' + message['content'] + '<|endoftext|>' + ('' if loop.last else '\n')}}"
    "{% endif %}"
    "{% endfor %}"
)


def setup_tokenizer(
    model_name_or_path, special_tokens=SPECIAL_TOKENS, chat_template=CHAT_TEMPLATE
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)
    tokenizer.add_special_tokens(
        {"eos_token": special_tokens.eos, "pad_token": special_tokens.pad}
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                special_tokens.system,
                special_tokens.user,
                special_tokens.assistant,
            ]
        }
    )
    if getattr(tokenizer, "add_bos_token", False) or getattr(
        tokenizer, "add_eos_token", False
    ):
        log_rank_0(
            "\033[91m!!!!!!!! tokenizer has add_bos_token or add_eos_token\033[0m"
        )
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    tokenizer.chat_template = chat_template
    return tokenizer


def get_sp_token(tokenizer, sp_string):
    sp_token = tokenizer.encode(sp_string, add_special_tokens=False)
    assert 1 == len(sp_token)
    return sp_token[0]
