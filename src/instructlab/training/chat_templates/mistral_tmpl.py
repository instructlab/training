# First Party
from instructlab.training.tokenizer_utils import SpecialTokens

SPECIAL_TOKENS = SpecialTokens(
    bos="<s>",
    eos="</s>",
    user="[INST]",
    assistant="[/INST]",
)

CHAT_TEMPLATE = (
    "{{ '<s>' }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'pretraining' %}"
    "{{ message['content'] + '</s>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + '</s>'}}"
    "{% endif %}"
    "{% endfor %}"
)
