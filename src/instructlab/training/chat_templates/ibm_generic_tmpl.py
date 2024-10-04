# SPDX-License-Identifier: Apache-2.0

# First Party
from instructlab.training.tokenizer_utils import SpecialTokens

SPECIAL_TOKENS = SpecialTokens(
    system="<|system|>",
    user="<|user|>",
    assistant="<|assistant|>",
    eos="<|endoftext|>",
    pad="<|pad|>",
)

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'pretraining' %}"
    "{{'<|pretrain|>' + message['content'] + '<|endoftext|>' + '<|/pretrain|>' }}"
    "{% elif message['role'] == 'system' %}"
    "{{'<|system|>'+ '\n' + message['content'] + '\n'}}"
    "{% elif message['role'] == 'user' %}"
    "{{'<|user|>' + '\n' + message['content'] + '\n'}}"
    "{% elif message['role'] == 'assistant' %}"
    "{{'<|assistant|>' + '\n' + message['content'] + '<|endoftext|>' + ('' if loop.last else '\n')}}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ '<|assistant|>' + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
)