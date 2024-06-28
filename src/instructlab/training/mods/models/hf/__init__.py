from . import llama
from . import mistral

def inject_padding_free_fa2():
    llama.inject_padding_free_fa2()
    mistral.inject_padding_free_fa2()