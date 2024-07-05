# First Party
from instructlab.training.tokenizer_utils import SpecialTokens

SPECIAL_TOKENS = SpecialTokens(
    system="<|system|>",
    user="<|user|>",
    assistant="<|assistant|>",
    eos="<|endoftext|>",
    pad="<|pad|>",
    contrastive_sep="<|contrastive_sep|>",
)

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'pretraining' %}"
    "{{'<|endoftext|>' + message['content'] + '<|endoftext|>'}}"
    "{% elif message['role'] == 'system' %}"
    "{{'<|system|>'+ '\n' + message['content'] + '\n'}}"
    "{% elif message['role'] == 'user' %}"
    "{% if 'rejected' in message %}"
    "{{'<|user|>' + '\n' + message['content'] + '\n' + '<|contrastive_sep|>' + <|user|>' + '\n' + message['rejected'] + '\n'}}"
    "{% else %}"
    "{{'<|user|>' + '\n' + message['content'] + '\n'}}"
    "{% endif %}"
    "{% elif message['role'] == 'assistant' %}"
    "{{'<|assistant|>' + '\n' + message['content'] + '<|endoftext|>' + ('' if loop.last else '\n')}}"
    "{% endif %}"
    "{% endfor %}"
)