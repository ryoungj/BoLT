import re

def sample_only(samples, shard_idx, input_key="text", output_key="latent", **kwargs) -> tuple[list[dict], dict]:
    outputs = []
    total_text_len = 0
    total_num_samples = 0
    for sample in samples:
        total_num_samples += 1
        text = sample[input_key]

        word_pattern = re.compile(r'\b\w+\b')
        text_len = len(word_pattern.findall(text))
        total_text_len += text_len
        sample[f"{input_key}_length"] = text_len
        
        outputs.append(sample)

    stats = {
        "total_num_samples": total_num_samples,
        f"total_{input_key}_length": total_text_len,
        f"avg_{input_key}_length": total_text_len / total_num_samples,
    }
    return outputs, stats
