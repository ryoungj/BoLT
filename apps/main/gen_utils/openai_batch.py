import json
import os
import time
import logging
from datetime import datetime
from openai import OpenAI
from tiktoken import encoding_for_model
from tenacity import retry, stop_after_attempt, wait_random_exponential
import dotenv
from tqdm import tqdm
import random
import numpy as np
from apps.main.gen_utils.misc_utils import get_logger
from apps.main.gen_utils.misc_utils import parse_json_file
from lingua.tokenizer import SpecialTokens

logger = get_logger(__name__)

httpx_logger = logging.getLogger("httpx")  # disable httpx logging
httpx_logger.setLevel(logging.WARNING)

dotenv.load_dotenv()


class OpenAIBatchClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)

    @retry(
        wait=wait_random_exponential(min=1, max=360), 
        stop=stop_after_attempt(20),
        reraise=True,
        before_sleep=lambda retry_state: logger.warning(
            f"Attempt {retry_state.attempt_number} failed with error: {retry_state.outcome.exception()}. Retrying..."
        )
    )
    def submit_batch_api_request(self, prompt_list: list[list[dict]], model: str, max_tokens: int = 1000, temperature: float = 0, start_idx: int = 0, filename_name: str = "batch_requests", tmp_dir: str = "./tmp"):
        os.makedirs(tmp_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = f"{tmp_dir}/{filename_name}_{timestamp}.jsonl"

        try:
            jsonl_data = []
            for i, prompt in enumerate(prompt_list):
                request_data = {
                    "custom_id": f"request-{start_idx+i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                }
                jsonl_data.append(request_data)

            with open(file_path, "w") as f:
                for entry in jsonl_data:
                    f.write(json.dumps(entry) + "\n")

            upload_file_response = self.client.files.create(
                file=open(file_path, "rb"),
                purpose="batch"
            )
            batch_input_file_id = upload_file_response.id
            logger.info(f"Uploaded batch input file with ID: {batch_input_file_id}")

            batch_response = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": filename_name
                }          
            )

            time.sleep(30)  # wait for the file to be uploaded?
            os.remove(file_path)
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
        return batch_response

    @retry(wait=wait_random_exponential(min=1, max=360), stop=stop_after_attempt(20))
    def check_batch_api_request(self, batch_id, allow_partial_results: bool = False):
        pbar = None
        while True:
            batch = self.client.batches.retrieve(batch_id)
            if batch.status == "in_progress":
                num_total = batch.request_counts.total
                if pbar is None:
                    pbar = tqdm(total=num_total, desc=f"Batch {batch_id} in progress")
                num_completed = batch.request_counts.completed
                # logger.info(f"Batch {batch_id} in progress: {num_completed}/{num_total} completed")
                if num_completed > pbar.n:
                    pbar.update(num_completed - pbar.n)
            if batch.status == "completed":
                logger.info(f"Batch {batch_id} completed")
                if pbar is not None:
                    pbar.close()
                return batch
            elif batch.status in ["failed", "cancelled", "expired", "cancelling"]:
                if not allow_partial_results:
                    raise Exception(f"Batch {batch_id} failed with status {batch.status} and error {batch.errors}")
                else:
                    logger.warning(f"Batch {batch_id} failed with status {batch.status} and error {batch.errors}")

                    if batch.status in ["cancelled", "expired", "cancelling"]:
                        logger.warning(f"Returning partial results for batch {batch_id} at status {batch.status}, {batch.request_counts.completed}/{batch.request_counts.total} completed")
                        return batch
            time.sleep(30)

    def get_batch_results(self, batch):
        result_file_id = batch.output_file_id
        content = self.client.files.content(result_file_id)
        for line in content.text.splitlines():
            yield json.loads(line)

    def delete_file(self, file_id):
        try:
            self.client.files.delete(file_id)
        except Exception as e:
            if "No such File object:" in str(e):
                pass
            else:
                logger.warning(f"Failed to delete file {file_id}: {e}")


def prepare_prompts(samples,input_key="text", output_key="latent", model="gpt-4o-mini", system_prompt=None, prompt_template="", prompt_placeholder_key="text", seed: int = 42):
    if not isinstance(prompt_template, list):
        prompt_template = [prompt_template]
    
    if not isinstance(system_prompt, list):
        system_prompt = [system_prompt]
    if len(system_prompt) == 1:
        system_prompt = [system_prompt[0]] * len(prompt_template)

    assert len(system_prompt) == len(prompt_template), "system_prompt and prompt_template must have the same length to be paired"

    outputs = []
    prompts = []
    prompt_lengths = {}
    encoding = encoding_for_model(model)

    warning_shown = False

    rng = np.random.default_rng(seed)

    for idx, sample in enumerate(samples):
        # prompt_idx = idx % len(prompt_template)
        prompt_idx = rng.integers(0, len(prompt_template))

        if isinstance(sample[input_key], dict):
            prompt_input = sample[input_key]
        else:
            prompt_input = {prompt_placeholder_key: sample[input_key]}
        
        user_prompt = prompt_template[prompt_idx].format(**prompt_input)
        if system_prompt[prompt_idx] is not None:
            if model.startswith("gpt"):
                prompt = [
                    {"role": "system", "content": system_prompt[prompt_idx]},
                    {"role": "user", "content": user_prompt},
                ]
            else:
                if not warning_shown:
                    logger.warning(f"Anthropic models do not support system prompts, appending it to the user prompt")
                    warning_shown = True
                prompt = [
                    {"role": "user", "content": system_prompt[prompt_idx] + "\n\n" + user_prompt},
                ]
        else:
            prompt = [
                {"role": "user", "content": user_prompt},
            ]

        if idx not in prompt_lengths:
            length = len(encoding.encode(prompt_template[prompt_idx]))
            if system_prompt[prompt_idx] is not None:
                length += len(encoding.encode(system_prompt[prompt_idx]))
            prompt_lengths[idx] = length
        
        sample["instruction_length"] = prompt_lengths[idx]

        prompts.append(prompt)
        outputs.append(sample)

    return prompts, outputs

def openai_batch_generation(
        samples, shard_idx, input_key="text", output_key="latent", model="gpt-4o-mini", 
        system_prompt=None, prompt_template="", prompt_placeholder_key="text", max_tokens: int = 1000, temperature: float = 0,
        replace_latent_start=None, replace_latent_end=None,
        max_retries=3, batch_size: int = 20000, tmp_dir: str = "./tmp",
        load_cached_batches: bool = True, delete_cloud_output_file: bool = False, seed: int = 42
    ):
    prompts, outputs = prepare_prompts(
        samples, input_key=input_key, output_key=output_key, model=model, 
        system_prompt=system_prompt, prompt_template=prompt_template, prompt_placeholder_key=prompt_placeholder_key, seed=seed
    )

    total_num_samples = len(prompts)
    client = OpenAIBatchClient()

    os.makedirs(tmp_dir, exist_ok=True)
    batch_tmp_file = os.path.join(tmp_dir, "submitted_batches.jsonl")
    prev_batches = []

    if load_cached_batches:
        if os.path.exists(batch_tmp_file):
            logger.info(f"Shard {shard_idx}: Loading cached batches from {batch_tmp_file}")
            with open(batch_tmp_file, "r") as f:
                for line in f:
                    prev_batches.append(json.loads(line.strip()))

    # loaded_run_batches = False
    # if reload_run_batches:   # FIXME: this is not working yet, seems that the output file cannot be retrieved
    #     if os.path.exists(batch_tmp_file):
    #         with open(batch_tmp_file, "r") as f:
    #             submitted_batch_ids = [json.loads(line.strip())["batch_id"] for line in f]
    #         submitted_batches = [client.check_batch_api_request(batch_id) for batch_id in submitted_batch_ids]
    #         logger.info(f"Shard {shard_idx}: Reloaded {len(submitted_batches)} batches from {batch_tmp_file}")
    #         loaded_run_batches = True
    #     else:
    #         logger.info(f"Shard {shard_idx}: No run batches found in {batch_tmp_file}, submitting new batches")

    def _submit_and_process_batches(_prompts, _outputs, _batch_size):
        submitted_batches = []
        for i in tqdm(range(0, len(_prompts), _batch_size), desc="Submitting batches"):
            requires_submit = True
            batch_idx = i // _batch_size
            if load_cached_batches:
                matched_batch_ids = list(map(lambda x: x["batch_id"], filter(lambda x: x["batch_idx"] == batch_idx and x["shard_idx"] == shard_idx, prev_batches)))
                if len(matched_batch_ids) > 0:
                    for matched_batch_id in matched_batch_ids:
                        retrived_batch = client.check_batch_api_request(matched_batch_id)
                        retrived_batch_output_file = retrived_batch.output_file_id

                        # check whether the file exists
                        if client.client.files.content(retrived_batch_output_file) is not None:
                            submitted_batches.append(retrived_batch)
                            logger.info(f"Shard {shard_idx} - batch {batch_idx}: found cached batch id {matched_batch_id}")
                            requires_submit = False
                            break
            
            if requires_submit:
                batch_prompts = _prompts[i:i+_batch_size]
                batch_response = client.submit_batch_api_request(batch_prompts, model, max_tokens=max_tokens, temperature=temperature, start_idx=i, filename_name=f"batch_requests_shard_{shard_idx}_batch_{batch_idx}", tmp_dir=tmp_dir)
                submitted_batches.append(batch_response)
                logger.info(f"Shard {shard_idx} - batch {batch_idx}: submitted batch id {batch_response.id}")

                batch_created_at = datetime.fromtimestamp(batch_response.created_at).strftime('%Y-%m-%d %H:%M:%S')
                with open(batch_tmp_file, "a") as f:
                    batch_json_obj = {
                        "batch_id": batch_response.id,
                        "created_at": batch_created_at,
                        "shard_idx": shard_idx,
                        "batch_idx": i // _batch_size,
                    }
                    f.write(json.dumps(batch_json_obj) + "\n")

        logger.info(f"Shard {shard_idx}: All batches submitted. Waiting for completion...")

        total_input_tokens = 0
        total_output_tokens = 0

        num_non_stop_finished = 0
        warning_shown = False

        sol_token = SpecialTokens.START_OF_LATENT.value
        eol_token = SpecialTokens.END_OF_LATENT.value

        for idx, batch_response in enumerate(submitted_batches):
            batch_id = batch_response.id

            completed_batch = client.check_batch_api_request(batch_id)
            results = client.get_batch_results(completed_batch)
            
            for result in results:
                output = result["response"]['body']['choices'][0]['message']['content']
                finish_reason = result["response"]['body']['choices'][0]['finish_reason']
                num_input_tokens = result["response"]["body"]['usage']['prompt_tokens']
                num_output_tokens = result["response"]["body"]['usage']['completion_tokens']
                custom_id = result["custom_id"]

                actual_idx = int(custom_id.split("-")[-1])

                logger.debug(f"Shard {shard_idx}: Output: {output}")
                logger.debug(f"Shard {shard_idx}: Finish reason: {finish_reason}")
                logger.debug(f"Shard {shard_idx}: Number of input tokens: {num_input_tokens}")
                logger.debug(f"Shard {shard_idx}: Number of output tokens: {num_output_tokens}")

                if finish_reason != "stop":
                    num_non_stop_finished += 1
                    if not warning_shown:
                        logger.warning(f"Shard {shard_idx}: Finished with reason {finish_reason} for request {result['id']} (custom_id: {custom_id})")
                        warning_shown = True

                if replace_latent_start is not None:
                    output = output.replace(replace_latent_start, sol_token)
                if replace_latent_end is not None:
                    output = output.replace(replace_latent_end, eol_token)

                sample = _outputs[actual_idx]
                sample[output_key] = output
                sample[f"{input_key}_length"] = num_input_tokens - sample.get("instruction_length", 0)
                sample[f"{output_key}_length"] = num_output_tokens

                total_input_tokens += num_input_tokens - sample.get("instruction_length", 0)
                total_output_tokens += num_output_tokens

            if num_non_stop_finished > 0:
                logger.warning(f"Shard {shard_idx}: {num_non_stop_finished} / {len(_outputs)} finished requests had non-stop finish reason")

            if delete_cloud_output_file:
                client.delete_file(completed_batch.output_file_id)
            client.delete_file(batch_response.input_file_id)

        return _outputs, total_input_tokens, total_output_tokens

    attempt = 0
    while True:
        try:
            outputs, total_input_tokens, total_output_tokens = _submit_and_process_batches(prompts, outputs, batch_size)

            stats = {
                "total_num_samples": total_num_samples,
                f"total_{input_key}_length": total_input_tokens,
                f"total_{output_key}_length": total_output_tokens,
                f"avg_{input_key}_length": total_input_tokens / total_num_samples,
                f"avg_{output_key}_length": total_output_tokens / total_num_samples,
                "compression_ratio": total_output_tokens / total_input_tokens,
            }
            return outputs, stats
        except Exception as e:
            logger.error(f"Shard {shard_idx}: attempt {attempt + 1} failed: {str(e)}")

            if "The batch input file is larger" in str(e):
                batch_size = batch_size // 2
                logger.warning(f"Shard {shard_idx}: Batch file is too large, splitting the batch size to {batch_size}")
            else:
                logger.info(f"Shard {shard_idx}: Retrying...")
                attempt += 1

            if attempt == max_retries - 1:
                raise Exception(f"Shard {shard_idx}: Failed to generate after {max_retries} attempts with error: {str(e)}")

if __name__ == "__main__":
    file_path = 'data/for_debug/dclm_baseline_sample_for_prompt_testing_num_10.jsonl'
    documents = parse_json_file(file_path)
    model = "gpt-4o-mini" 
    system_prompt = "You are an expert in information extraction and summarization."
    # prompt_template = "Summarize the key facts and claims in the following text using very simple language that's understandable to a smart middle schooler. Be concise and use at most four sentences. Write in a direct style without using phrases like 'The text discusses' or 'The author thinks'.\n\nText: {text}"
    # prompt_template = "Text: {text}\n\nPlease break down the preceding text into a numbered list of atomic facts. Only include notable facts and use words a kid could understand. No decorative language or special formatting."
    prompt_template = "Text: {text}\n\nFor the preceding text give me a short paraphrase of the same in high quality English language as in sentences on Wikipedia. Write in a direct style without using phrases like 'The text discusses' or 'The author thinks'."
    
    results, stats = openai_batch_generation(documents, 0, model=model, system_prompt=system_prompt, prompt_template=prompt_template, tmp_dir="./tmp/debug", reload_run_batches=False)

    for r in results:
        print("\n" * 2)
        print("=" * 50, "text", "=" * 50)
        print(r["text"])
        print("\n" * 2)
        print("=" * 50, "latent", "=" * 50)
        print(r["latent"])
    print(stats)