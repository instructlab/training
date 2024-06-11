from dataclasses import dataclass, field
from transformers import AutoTokenizer

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
    "{% if message['role'] == 'pretraining' %}"
    "{{'<|endoftext|>' + message['content'] + '<|endoftext|>'}}"
    "{% elif message['role'] == 'system' %}"
    "{{'<|system|>'+ '\n' + message['content'] + '\n'}}"
    "{% elif message['role'] == 'user' %}"
    "{{'<|user|>' + '\n' + message['content'] + '\n'}}"
    "{% elif message['role'] == 'assistant' %}"
    "{{'<|assistant|>' + '\n' + message['content'] + '<|endoftext|>' + ('' if loop.last else '\n')}}"
    "{% endif %}"
    "{% endfor %}"
)


def setup_tokenizer(
    model_name_or_path, SPECIAL_TOKENS=SPECIAL_TOKENS, CHAT_TEMPLATE=CHAT_TEMPLATE
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)
    tokenizer.add_special_tokens(
        {"eos_token": SPECIAL_TOKENS.eos, "pad_token": SPECIAL_TOKENS.pad}
    )
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                SPECIAL_TOKENS.system,
                SPECIAL_TOKENS.user,
                SPECIAL_TOKENS.assistant,
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

    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def get_sp_token(tokenizer, sp_string):
    sp_token = tokenizer.encode(sp_string, add_special_tokens=False)
    assert 1 == len(sp_token)
    return sp_token[0]
