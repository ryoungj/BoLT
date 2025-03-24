# Copyright 2024 Neil Band
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Light wrapper for OpenAI API.

Reference API:
    https://beta.openai.com/docs/api-reference/completions/create

Internal map:
    https://github.com/lxuechen/ml-swissknife/blob/main/ml_swissknife/openai_utils.py
"""
import copy
import functools
import math
import multiprocessing
import openai
import os
import random
import sys
import time
import tqdm
# from openai import openai_object
from typing import Optional, Sequence, Union

import logging

from apps.main.gen_utils.interface_utils import DecodingArguments, PromptMessage, Response, requires_chatml, prompt_to_chatml


openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


def _openai_completion_helper(
    prompt_batch: Sequence[PromptMessage],
    is_chat: bool,
    sleep_time: int,
    openai_organization_ids: Optional[Sequence[str]] = None,
    openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY", None),
    **shared_kwargs,
) -> Sequence[Response]:
    if openai_api_key is not None:
        openai.api_key = openai_api_key

    # randomly select orgs
    if openai_organization_ids is not None:
        openai.organization = random.choice(openai_organization_ids)

    client = openai.OpenAI(timeout=180, max_retries=1)

    # copy shared_kwargs to avoid modifying it
    shared_kwargs = copy.deepcopy(shared_kwargs)

    while True:
        try:
            if is_chat:
                completion_batch = client.chat.completions.create(messages=prompt_batch[0], **shared_kwargs)
            else:
                completion_batch = client.completions.create(prompt=prompt_batch, **shared_kwargs)
            
            choices = completion_batch.choices
            responses = []
            for choice in choices:
                assert choice.message.role == "assistant"
                response = Response(
                    text=choice.message.content,
                    prompt_tokens=completion_batch.usage.prompt_tokens / len(prompt_batch),
                    completion_tokens=completion_batch.usage.completion_tokens / len(prompt_batch),
                    total_tokens=completion_batch.usage.total_tokens / len(prompt_batch),
                    finish_reason=choice.finish_reason,
                    logprobs=choice.logprobs,

                )
                responses.append(response)
            break
        except openai.OpenAIError as e:
            logging.warning(f"OpenAIError: {str(e)}.")
            if "Please reduce your prompt" in str(e):
                shared_kwargs["max_tokens"] = int(shared_kwargs["max_tokens"] * 0.8)
                logging.warning(f"Reducing target length to {shared_kwargs['max_tokens']}, Retrying...")
            else:
                logging.warning(f"Sleeping for {sleep_time} seconds and retrying...")
                if openai_organization_ids is not None and len(openai_organization_ids) > 1:
                    openai.organization = random.choice(
                        [o for o in openai_organization_ids if o != openai.organization]
                    )
                    logging.warning(f"Switching to organization: {openai.organization} for OAI API key.")
                time.sleep(sleep_time)  # Annoying rate limit on requests.
                sleep_time = min(sleep_time * 2, 300)
    return responses


def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: DecodingArguments,
    model_name="text-davinci-003",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    num_procs=1,
    **decoding_kwargs,
) -> Union[Response, Sequence[Response], Sequence[Sequence[Response]],]:
    """Decode with OpenAI API.

    Args:
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    logging.info(f"Decoding with OpenAI API model {model_name} and numproc == {num_procs}.")
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    # convert prompts to chat format
    is_chat = requires_chatml(model_name)
    is_chat_format = isinstance(prompts[0][0], dict)
    if is_chat:
        if batch_size > 1:
            logging.warning("batch_size > 1 is not supported yet for chat models. Setting to 1")
            batch_size = 1
        if not is_chat_format:
            prompts = [prompt_to_chatml(prompt) for prompt in prompts]

    if max_batches < sys.maxsize:
        logging.warning(
            "`max_batches` will be deprecated in the future, please use `max_instances` instead."
            "Setting `max_instances` to `max_batches * batch_size` for now."
        )
        max_instances = max_batches * batch_size

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    shared_kwargs = dict(
        model=model_name,
        **decoding_args.__dict__,
    )
    shared_kwargs.update(decoding_kwargs)  # override default arguments if specified

    if num_procs == 1:
        completions = []
        for prompt_batch in tqdm.tqdm(prompt_batches, desc="prompt_batches", total=len(prompt_batches)):
            completions.append(_openai_completion_helper(prompt_batch, is_chat=is_chat, sleep_time=sleep_time, **shared_kwargs))
    else:
        with multiprocessing.Pool(num_procs) as p:
            partial_completion_helper = functools.partial(
                _openai_completion_helper, sleep_time=sleep_time, is_chat=is_chat, **shared_kwargs
            )
            completions = list(
                tqdm.tqdm(
                    p.imap(partial_completion_helper, prompt_batches),
                    desc="prompt_batches",
                    total=len(prompt_batches),
                )
            )
    # flatten the list
    completions = [completion for completion_batch in completions for completion in completion_batch]

    if return_text:
        completions = [completion.text for completion in completions]
    if decoding_args.n > 1:
        # make completions a nested list, where each entry is a consecutive decoding_args.n of original entries.
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        # Return non-tuple if only 1 input and 1 generation.
        (completions,) = completions
    return completions