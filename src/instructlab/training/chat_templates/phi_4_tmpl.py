# SPDX-License-Identifier: Apache-2.0

# First Party
from instructlab.training.chat_templates.utils import SpecialTokens, TokenInfo

SPECIAL_TOKENS = SpecialTokens(
    start_role=TokenInfo("<|im_start|>", add_to_tokenizer=False),
    end_role=TokenInfo("<|im_sep|>", add_to_tokenizer=False),
    #tool=TokenInfo("<|tool_call|>", add_to_tokenizer=True),
    eos=TokenInfo("<|im_end|>", add_to_tokenizer=False),
    bos=TokenInfo("<|endoftext|>", add_to_tokenizer=False),
    pad=TokenInfo("<|dummy_85|>", add_to_tokenizer=False),
)


CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if (message['role'] == 'system') %}"
    "{{'<|im_start|>system<|im_sep|>' + message['content'] + '<|im_end|>'}}"
    "{% elif (message['role'] == 'user') %}"
    "{{'<|im_start|>user<|im_sep|>' + message['content'] + '<|im_end|>'}}"
    "{% elif (message['role'] == 'assistant') %}"
    "{{'<|im_start|>assistant<|im_sep|>' + message['content'] + '<|im_end|>'}}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant<|im_sep|>' }}"
    "{% endif %}"
)
  