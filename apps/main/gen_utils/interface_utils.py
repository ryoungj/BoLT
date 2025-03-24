import dataclasses
from typing import Optional, Sequence, List, Union


@dataclasses.dataclass
class DecodingArguments(object):
    """Follows OpenAI API decoding arguments."""

    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    # Heuristic stop when about to generate next function.
    # stop: Optional[Tuple[str, ...]] = ("}\n\nstatic", "}\n\n/*")
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # If you need these, pass them in as decoding_kwargs.
    # best_of: int = 1
    # logit_bias: dict = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None


@dataclasses.dataclass
class Response(object):
    text: str
    prompt_tokens: int = None
    completion_tokens: int = None
    total_tokens: int = None
    finish_reason: str = None
    logprobs: Optional[int] = None
    metadata: Optional[dict] = None

@dataclasses.dataclass
class ChatMessage(object):
    role: str
    content: str

PromptMessage = Union[str, List[ChatMessage]]

def requires_chatml(model: str) -> bool:
    """Whether a model requires the ChatML format."""
    # TODO: this should ideally be an OpenAI function... Maybe it already exists?
    return "turbo" in model or "gpt-4" in model or "claude-3" in model


def string_to_dict(to_convert):
    """Converts a string with equal signs to dictionary. E.g.
    >>> string_to_dict(" name=user university=stanford")
    {'name': 'user', 'university': 'stanford'}
    """
    return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in to_convert.split(" ") if len(s) > 0}


def prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>") -> PromptMessage:
    """Convert a text prompt to ChatML formal

    Examples
    --------
    >>> prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>system
    name=example_user\nKnock knock.\n<|im_end|>\n<|im_start|>system name=example_assistant\nWho's
    there?\n<|im_end|>\n<|im_start|>user\nOrange.\n<|im_end|>"
    >>> print(prompt)
    <|im_start|>system
    You are a helpful assistant.
    <|im_end|>
    <|im_start|>system name=example_user
    Knock knock.
    <|im_end|>
    <|im_start|>system name=example_assistant
    Who's there?
    <|im_end|>
    <|im_start|>user
    Orange.
    <|im_end|>
    >>> prompt_to_chatml(prompt)
    [{'role': 'system', 'content': 'You are a helpful assistant.'},
     {'role': 'user', 'content': 'Knock knock.'},
     {'role': 'assistant', 'content': "Who's there?"},
     {'role': 'user', 'content': 'Orange.'}]
    """
    prompt = prompt.strip()
    assert prompt.startswith(start_token)
    assert prompt.endswith(end_token)

    message = []
    for p in prompt.split("<|im_start|>")[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            # based on https://github.com/openai/openai-cookbook/blob/main/examples
            # /How_to_format_inputs_to_ChatGPT_models.ipynb
            # and https://github.com/openai/openai-python/blob/main/chatml.md it seems that system can specify a
            # dictionary of other args
            other_params = string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message