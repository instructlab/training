# First Party
from instructlab.training.tokenizer_utils import SpecialTokens, TokenInfo

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
    "{% endfor %}"
)
