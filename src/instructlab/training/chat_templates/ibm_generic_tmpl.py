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

'''
format of a sample would be:
{
    "messages": [
        {"role": "system", "content": "Hello, how can I help you today?"},
        {"role": "user", "content": "I'm looking for a restaurant."},
        {"role": "assistant", "content": "Sure! What type of cuisine are you looking for?"},
        {"role": "user", "content": "I'm in the mood for Italian."},
        {"role": "assistant", "content": "Great! I recommend trying out Pasta Palace."},
        {"role": "user", "content": "Sounds good! Where is it located?"},
        {"role": "assistant", "content": "It's located at 123 Main Street."},
        {"role": "user", "content": "Thanks! I'll check it out."},
        {"role": "assistant", "content": "You're welcome! Enjoy your meal!"},
    ]
}
and for pretraining:
{
    "messages": [
        {"role": "pretraining", "content": "Hello, how can I help you today?"},
    ]
}
'''
