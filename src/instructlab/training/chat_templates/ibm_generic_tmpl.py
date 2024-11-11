# SPDX-License-Identifier: Apache-2.0

# First Party
from instructlab.training.chat_templates.utils import SpecialTokens, TokenInfo

SPECIAL_TOKENS = SpecialTokens(
    start_role=TokenInfo("<|start_of_role|>", add_to_tokenizer=True),
    end_role=TokenInfo("<|end_of_role|>", add_to_tokenizer=True),
    tool=TokenInfo("<|tool_call|>", add_to_tokenizer=True),
    eos=TokenInfo("<|end_of_text|>", add_to_tokenizer=True),
    bos=TokenInfo("<|end_of_text|>", add_to_tokenizer=True),
    pad=TokenInfo("<|end_of_text|>", add_to_tokenizer=True),
)

CHAT_TEMPLATE = (
    "{%- if tools %}"
    "{{ '<|start_of_role|>available_tools<|end_of_role|>\n' }}"
    "{% for tool in tools %}"
    "{{ tool | tojson(indent=4) }}"
    "{% if not loop.last %}"
    "{{- '\n\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{{ '<|end_of_text|>\n' }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|start_of_role|>system<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}"
    "{% elif message['role'] == 'pretraining' %}"
    "{{ '<|pretrain|>' + message['content'] + '<|end_of_text|>' + '<|/pretrain|>'}}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|start_of_role|>user<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|start_of_role|>assistant<|end_of_role|>'  + message['content'] + '<|end_of_text|>\n' }}"
    "{% elif message['role'] == 'assistant_tool_call' %}"
    "{{ '<|start_of_role|>assistant<|end_of_role|><|tool_call|>' + message['content'] + '<|end_of_text|>\n' }}"
    "{% elif message['role'] == 'tool_response' %}"
    "{{ '<|start_of_role|>tool_response<|end_of_role|>' + message['content'] + '<|end_of_text|>\n' }}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ '<|start_of_role|>assistant<|end_of_role|>' }}"
    "{% endif %}"
    "{% endfor %}"
)
