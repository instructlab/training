# First Party
from instructlab.training.tokenizer_utils import SpecialTokens

SPECIAL_TOKENS = SpecialTokens(
    bos="<s>",
    eos="</s>",
    user="[INST]",
    assistant="[/INST]",
    contrastive_sep="<|contrastive_sep|>",
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
    "{% elif message['role'] == 'assistant_w_rejected' %}"
    "{{ message['content'] + '</s>' }}"
    "{% for rejected_message in message['rejected'] %}"
    "{{ '<|contrastive_sep|>' + rejected_message + '</s>' }}"
    "{% endfor %}"
    "{% endif %}"
    "{% endfor %}"
)
