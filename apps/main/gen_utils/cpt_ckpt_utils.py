LLAMA_PRETRAINED_CKPT_NAMES = {
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-405B",
    "TinyLlama/TinyLlama_v1.1",
}


def is_llama_pretrained_ckpt(ckpt_path: str) -> bool:
    return any(name in str(ckpt_path) for name in LLAMA_PRETRAINED_CKPT_NAMES)