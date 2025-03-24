# Copyright 2024 Lingua-Fork and the Tatsu Lab.
# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gc
import json
import os
import tempfile
import shutil

import torch
from transformers import AutoTokenizer, GenerationConfig, LlamaConfig, LlamaForCausalLM

from apps.main.transformer import LMTransformer, LMTransformerArgs
from pathlib import Path

import torch

from omegaconf import OmegaConf

from apps.main.transformer import LMTransformer, LMTransformerArgs
from lingua.args import dataclass_from_dict
from lingua.checkpoint import CONSOLIDATE_NAME



"""
TODO(@nband): update docstring below

Sample usage:

```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 1B --llama_version 3.2 --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).

If you want your tokenizer to add a bos automatically you should update the tokenizer._tokenizers.post_processor:

```py
from tokenizers import processors
bos = "<|begin_of_text|>"
tokenizer._tokenizers.post_processor = processors.Sequence(
    [
        processors.ByteLevel(trim_offsets=False),
        processors.TemplateProcessing(
            single=f"{bos}:0 $A:0",
            pair=f"{bos}:0 $A:0 {bos}:1 $B:1",
            special_tokens=[
                (bos, tokenizer.encode(bos)),
            ],
        ),
    ]
)
```
"""

def load_consolidated_model(
    consolidated_path,
    model_cls=LMTransformer,
    model_args_cls=LMTransformerArgs,
    dtype=torch.bfloat16,
):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)
    model_args = dataclass_from_dict(model_args_cls, config, strict=False)
    model = model_cls(model_args)
    st_dict = torch.load(ckpt_path / CONSOLIDATE_NAME, weights_only=True)

    if "model" in st_dict:
        model.load_state_dict(st_dict["model"])
    else:
        model.load_state_dict(st_dict)

    if model_args.rope_scaling is not None:
        # Init rope embeddings with RoPE scaling config
        model.rope_embeddings.reset_parameters()

    model = model.cuda().eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=dtype)
        
    return model


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def write_model(
    model_path,
    input_base_path,
    tokenizer_path,
    safe_serialization=True,
    push_to_hub=False,
):
    print("Converting the model.")
    params = read_json(os.path.join(input_base_path, "params.json"))
    params = params.get("model", params)
    n_layers = params["n_layers"]
    n_heads = params["n_heads"]
    dim = params["dim"]
    dims_per_head = dim // n_heads
    base = params["rope_theta"]
    assert base is not None, "Rope theta is not set"
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

    max_position_embeddings = params['max_seqlen']
    assert max_position_embeddings is not None, "Max position embeddings is not set"

    # For now, don't support rope_scaling (if we need this, @nband's script supports)
    assert params['rope_scaling'] is None, "Rope scaling is not supported yet"

    if params.get("n_kv_heads", None) is not None:
        num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        key_value_dim = dims_per_head * num_key_value_heads
    else:  # compatibility with other checkpoints
        num_key_value_heads = n_heads
        key_value_dim = dim

    # permute for sliced rotary
    def permute(w, n_heads, dim1=dim, dim2=dim):
        return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)
    
    # with tempfile.TemporaryDirectory() as tmp_model_path:
    tmp_model_path = tempfile.mkdtemp()
    try:
        if os.path.exists(os.path.join(model_path, "success.txt")):
            print(f"Model already converted at {model_path}")
            return

        print(f"Fetching all parameters from the checkpoint at {input_base_path}.")

        # Load weights
        # Not sharded
        model = load_consolidated_model(input_base_path)
        loaded = model.cpu().state_dict()

        param_count = 0
        index_dict = {"weight_map": {}}
        for layer_i in range(n_layers):
            filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
            
            # Unsharded
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wq.weight"], n_heads=n_heads
                ),
                f"model.layers.{layer_i}.self_attn.k_proj.weight": permute(
                    loaded[f"layers.{layer_i}.attention.wk.weight"],
                    n_heads=num_key_value_heads,
                    dim1=key_value_dim,
                ),
                f"model.layers.{layer_i}.self_attn.v_proj.weight": loaded[f"layers.{layer_i}.attention.wv.weight"],
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[f"layers.{layer_i}.attention.wo.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w1.weight"],
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w2.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": loaded[f"layers.{layer_i}.feed_forward.w3.weight"],
                f"model.layers.{layer_i}.input_layernorm.weight": loaded[
                    f"layers.{layer_i}.attention_norm.weight"
                ],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[
                    f"layers.{layer_i}.ffn_norm.weight"
                ],
            }
            
            state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq
            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()

            torch.save(state_dict, os.path.join(tmp_model_path, filename))

        filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
        
        # Unsharded
        state_dict = {
            "model.embed_tokens.weight": loaded["tok_embeddings.weight"],
            "model.norm.weight": loaded["norm.weight"],
            "lm_head.weight": loaded["output.weight"],
        }

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()

        torch.save(state_dict, os.path.join(tmp_model_path, filename))

        # Write configs
        index_dict["metadata"] = {"total_size": param_count * 2}
        write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
        # ffn_dim_multiplier = params["ffn_dim_multiplier"] if "ffn_dim_multiplier" in params else 1
        # multiple_of = params["multiple_of"] if "multiple_of" in params else 256

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        assert bos_token_id is not None, "Bos token id is not set"
        assert eos_token_id is not None, "Eos token id is not set"
        vocab_size = len(tokenizer)
        assert vocab_size == params['vocab_size'] == state_dict['lm_head.weight'].shape[0] == state_dict['model.embed_tokens.weight'].shape[0], (
            "Found inconsistent vocab size between tokenizer, params, and state_dict: "
            f"tokenizer: {vocab_size}, params: {params['vocab_size']}, "
            f"lm_head: {state_dict['lm_head.weight'].shape[0]}, "
            f"embed_tokens: {state_dict['model.embed_tokens.weight'].shape[0]}"
        )

        # if use_rope_scaling:
        #     rope_scaling = {
        #         "factor": 32.0,
        #         "low_freq_factor": 1.0,
        #         "high_freq_factor": 4.0,
        #         "original_max_position_embeddings": 8192,
        #         "rope_type": "llama3",
        #     }
        # else:

        config = LlamaConfig(
            hidden_size=dim,
            intermediate_size=params["hidden_dim"],
            num_attention_heads=params["n_heads"],
            num_hidden_layers=params["n_layers"],
            rms_norm_eps=params["norm_eps"],
            num_key_value_heads=num_key_value_heads,
            vocab_size=vocab_size,
            rope_theta=base,
            rope_scaling=None,
            max_position_embeddings=max_position_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=False
        )

        config.save_pretrained(tmp_model_path)

        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        generation_config.save_pretrained(tmp_model_path)

        # Make space so we can load the model properly now.
        del state_dict
        del loaded
        gc.collect()

        print("Loading the checkpoint in a Llama model.")
        model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

        # Avoid saving this as part of the config.
        del model.config._name_or_path
        model.config.torch_dtype = torch.bfloat16

        print("Saving in the Transformers format.")
        if push_to_hub:
            print("Pushing to the hub.")
            model.push_to_hub(model_path, safe_serialization=safe_serialization, private=True, use_temp_dir=True)
        else:
            print("Saving to disk.")
            model.save_pretrained(model_path, safe_serialization=safe_serialization)

        # Add a text file indicating success
        with open(os.path.join(model_path, "success.txt"), "w") as f:
            f.write("Successfully converted Lingua Llama weights to HF format.")
    finally:
        # Clean up the temporary directory
        if os.path.exists(tmp_model_path):
            shutil.rmtree(tmp_model_path, ignore_errors=True)

    print("Done!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of consolidated Lingua Llama weights, e.g., /path/to/checkpoints/0000040000/consolidated",
    )
    parser.add_argument(
        "--tokenizer_path",
        help="Location of the HF-format tokenizer, e.g., /path/to/tokenizer. "
             "For now only used to get bos_token_id and eos_token_id and confirm consistency of vocab size.",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model",
    )
    parser.add_argument(
        "--push_to_hub",
        help="Whether or not to push the model to the hub at `output_dir` instead of saving it locally.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--safe_serialization", action="store_true", default=True, help="Whether or not to save using `safetensors`."
    )
    args = parser.parse_args()

    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        tokenizer_path=args.tokenizer_path,
        safe_serialization=args.safe_serialization,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()