# Standard
from dataclasses import dataclass, field
from typing import List


@dataclass
class TokenInfo:
    token: str
    add_to_tokenizer: bool = False


@dataclass
class SpecialTokens:
    system: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    user: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    assistant: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    eos: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    pad: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    bos: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    start_role: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    end_role: TokenInfo = field(default_factory=lambda: TokenInfo(""))
    tool: TokenInfo = field(default_factory=lambda: TokenInfo(""))

    def get_tokens_to_add(self) -> List[str]:
        return [
            token_info.token
            for token_info in self.__dict__.values()
            if token_info.add_to_tokenizer and token_info.token
        ]
