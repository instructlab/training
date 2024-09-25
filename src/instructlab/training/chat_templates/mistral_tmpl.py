# SPDX-License-Identifier: Apache-2.0

# First Party
from instructlab.training.tokenizer_utils import SpecialTokens, TokenInfo

SPECIAL_TOKENS = SpecialTokens(
    bos=TokenInfo("<s>", add_to_tokenizer=True),
    eos=TokenInfo("</s>", add_to_tokenizer=True),
    user=TokenInfo("[INST]", add_to_tokenizer=False),
    assistant=TokenInfo("[/INST]", add_to_tokenizer=False),
)

CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{%- set system_message = messages[0]['content'] %}"
    "{%- set loop_messages = messages[1:] %}"
    "{%- else %}"
    "{%- set loop_messages = messages %}"
    "{%- endif %}"
    "{{ '<s>' }}"
    "{%- for message in loop_messages %}"
    "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{%- endif %}"
    "{%- if message['role'] == 'user' %}"
    "{%- if loop.first and system_message is defined %}"
    "{{- ' [INST] ' + system_message + '\n\n' + message['content'] + ' [/INST]' }}"
    "{%- else %}"
    "{{- ' [INST] ' + message['content'] + ' [/INST]' }}"
    "{%- endif %}"
    "{%- elif message['role'] == 'pretraining' %}"
    "{{- '<|pretrain|>' + message['content'] + '</s>' + '<|/pretrain|>' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- ' ' + message['content'] + '</s>'}}"
    "{%- else %}"
    "{{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}"
    "{%- endif %}"
    "{%- endfor %}"
)
