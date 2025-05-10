# SPDX-License-Identifier: Apache-2.0

# First Party
from instructlab.training.chat_templates.utils import SpecialTokens, TokenInfo

SPECIAL_TOKENS = SpecialTokens(
    system=TokenInfo("<|system|>", add_to_tokenizer=True),
    user=TokenInfo("<|user|>", add_to_tokenizer=True),
    assistant=TokenInfo("<|assistant|>", add_to_tokenizer=True),
    eos=TokenInfo("<|endoftext|>", add_to_tokenizer=True),
    pad=TokenInfo("<|pad|>", add_to_tokenizer=True),
    bos=TokenInfo("<|begginingoftext|>", add_to_tokenizer=True),
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
