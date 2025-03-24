"""Synchronous generation utilities. Mostly used for quick debugging and iteration."""

from tiktoken import encoding_for_model
from dotenv import load_dotenv

load_dotenv()

from apps.main.gen_utils.openai_utils import openai_completion
from apps.main.gen_utils.anthropic_utils import anthropic_completion
from apps.main.gen_utils.interface_utils import DecodingArguments
from apps.main.gen_utils.openai_batch import prepare_prompts

from apps.main.gen_utils.misc_utils import get_logger
from apps.main.gen_utils.misc_utils import parse_json_file
from lingua.tokenizer import SpecialTokens

logger = get_logger(__name__)


def generate_sync(
        samples, shard_idx, input_key="text", output_key="latent", model="gpt-4o-mini", 
        system_prompt=None, prompt_template="", prompt_placeholder_key="text", 
        max_tokens: int = 1000, temperature: float = 0,
        replace_latent_start=None, replace_latent_end=None,
        compute_logprobs=False, num_procs=1, seed: int = 42, **kwargs
    ):

    prompts, outputs = prepare_prompts(
        samples, input_key=input_key, output_key=output_key, model=model, 
        system_prompt=system_prompt, prompt_template=prompt_template, prompt_placeholder_key=prompt_placeholder_key, seed=seed
    )

    decoding_args = DecodingArguments(max_tokens=max_tokens, temperature=temperature, logprobs=compute_logprobs)
    
    if model.startswith("gpt"):
        results = openai_completion(prompts, decoding_args, model_name=model, num_procs=num_procs, **kwargs)
    elif model.startswith("claude"):
        if compute_logprobs:
            raise ValueError("Claude models do not support logprobs")
        results = anthropic_completion(prompts, decoding_args, model_name=model, num_procs=num_procs, **kwargs)
    else:
        raise ValueError(f"Model {model} not supported")
    
    total_num_samples = len(prompts)
    total_input_tokens = 0
    total_output_tokens = 0

    sol_token = SpecialTokens.START_OF_LATENT.value
    eol_token = SpecialTokens.END_OF_LATENT.value

    for idx, result in enumerate(results):
        sample = outputs[idx]
        
        output = result.text
        if replace_latent_start is not None:
            output = output.replace(replace_latent_start, sol_token)
        if replace_latent_end is not None:
            output = output.replace(replace_latent_end, eol_token)

        sample[output_key] = output
        sample[f"{input_key}_length"] = result.prompt_tokens - sample.get("instruction_length", 0)
        sample[f"{output_key}_length"] = result.completion_tokens

        if compute_logprobs:
            output_logprobs = result.logprobs.content
            # print([(t.logprob, t.token) for t in output_logprobs])
            sum_logprobs = sum([t.logprob for t in output_logprobs])
            sample[f"{output_key}_logprobs"] = sum_logprobs
            sample[f"{output_key}_avg_logprob"] = sum_logprobs / result.completion_tokens

        total_input_tokens += result.prompt_tokens - sample.get("instruction_length", 0)
        total_output_tokens += result.completion_tokens


    stats = {
        "total_num_samples": total_num_samples,
        f"total_{input_key}_length": total_input_tokens,
        f"total_{output_key}_length": total_output_tokens,
        f"avg_{input_key}_length": total_input_tokens / total_num_samples,
        f"avg_{output_key}_length": total_output_tokens / total_num_samples,
        "compression_ratio": total_output_tokens / total_input_tokens,
    }

    return outputs, stats



if __name__ == "__main__":
    file_path = 'data/for_debug/dclm_baseline_sample_for_prompt_testing_num_10.jsonl'
    documents = parse_json_file(file_path)
    model = "gpt-4o-mini" 
    # model = "claude-3-haiku-20240307" 
    system_prompt = "You are an expert in information extraction and summarization."
    prompt_template = "Text: {text}\n\nFor the preceding text give me a short paraphrase of the same in high quality English language as in sentences on Wikipedia. Write in a direct style without using phrases like 'The text discusses' or 'The author thinks'."
    
    results, stats = generate_sync(
        documents, 0, model=model, system_prompt=system_prompt, prompt_template=prompt_template, num_procs=5,
        compute_logprobs=True, temperature=0.0,
    )

    for r in results:
        print("\n" * 2)
        # print("=" * 50, "text", "=" * 50)
        # print(r["text"])
        # print("\n" * 2)
        print("=" * 50, "latent", "=" * 50)
        print(r["latent"])
        print("=" * 50, "logprobs", "=" * 50)
        print(r["latent_logprobs"], r["latent_avg_logprob"])
    print(stats)
