import argparse
import re
from pathlib import Path
import os

import torch
from torch.distributed.checkpoint.format_utils import torch_save_to_dcp
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

from lingua.tokenizer import hf_add_special_tokens, SpecialTokens

def permute(w, n_heads, dim1=2048, dim2=2048):
    """Permute the qk weights from LLama3 -> HF
    From https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py#L221
    """
    return w.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def inverse_permute(w, n_heads, dim1=2048, dim2=2048):
    """Inverse permute the qk weights from HF -> LLama3"""
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def download_and_convert_hf_to_dcp(hf_model_name: str, output_dir: str, dtype: str = "float32", add_special_tokens: bool = True):
    """
    Downloads a HuggingFace model and converts it to DCP format for Lingua.
    Args:
        hf_model_name: Name/path of the HuggingFace model
        output_dir: Directory to save the converted checkpoint
        dtype: Data type to load the model in ("float16", "bfloat16", or "float32")
    """
    
    consolidated_path = os.path.join(output_dir, "consolidated/consolidated.pth")
    os.makedirs(os.path.dirname(consolidated_path), exist_ok=True)

    final_state_dict = None
    # Write the tokenizer to the output directory
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    if add_special_tokens:
        tokenizer = hf_add_special_tokens(tokenizer, SpecialTokens)
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    
    if hf_model_name.startswith("meta-llama"):
        hf_hub_download(repo_id=hf_model_name, filename="original/consolidated.00.pth", local_dir=output_dir)
        os.rename(os.path.join(output_dir, "original/consolidated.00.pth"), consolidated_path)
        os.rmdir(os.path.join(output_dir, "original"))
    elif hf_model_name.startswith("TinyLlama"):
        print("Loading model and its config from HuggingFace...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name, torch_dtype=dtype, trust_remote_code=True
        )

        state_dict = model.state_dict()

        rename_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.o_proj.bias": "layers.{}.attention.wo.bias",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        config = model.config
        hidden_dim = config.hidden_size
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads

        final_state_dict = {}
        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = rename_map[abstract_key]
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = rename_map[key]

            if "wq" in new_key:
                value = inverse_permute(value, n_heads=n_heads, dim1=hidden_dim, dim2=hidden_dim)
            elif "wk" in new_key:
                value = inverse_permute(value, n_heads=n_kv_heads, dim1=hidden_dim // n_heads * n_kv_heads, dim2=hidden_dim)

            final_state_dict[new_key] = value
    else:
        os.rmdir(output_dir)
        raise ValueError(f"Unsupported model: {hf_model_name}")


    if add_special_tokens:
        if final_state_dict is None:
            final_state_dict = torch.load(consolidated_path)

        new_num_embeddings = len(tokenizer)
        old_num_embeddings = final_state_dict["tok_embeddings.weight"].shape[0]
        model_dim = final_state_dict["tok_embeddings.weight"].shape[1]

        if new_num_embeddings != old_num_embeddings:
            print(f"Resizing embeddings from {old_num_embeddings} to {new_num_embeddings}...")
        
            weight_tying = torch.allclose(final_state_dict["output.weight"], final_state_dict["tok_embeddings.weight"])
            print(f"Weight tying: {weight_tying}")

            torch.manual_seed(42)
            init_std = model_dim ** (-0.5)

            def _init_embeddings(embeddings, std):
                embeddings.normal_(0, std)
                with torch.no_grad():
                    embeddings.clamp_(-3 * std, 3 * std)

                return embeddings

            orig_tok_embeddings = final_state_dict["tok_embeddings.weight"]
            new_tok_embeddings = _init_embeddings(torch.empty(new_num_embeddings - old_num_embeddings, model_dim, dtype=orig_tok_embeddings.dtype, device=orig_tok_embeddings.device, requires_grad=orig_tok_embeddings.requires_grad), init_std)
            
            final_state_dict["tok_embeddings.weight"] = torch.cat([
                orig_tok_embeddings, 
                new_tok_embeddings
            ], dim=0)

            if weight_tying:
                final_state_dict["output.weight"] = final_state_dict["tok_embeddings.weight"]
            else:
                orig_output_weight = final_state_dict["output.weight"]
                new_output_weight = _init_embeddings(torch.empty(new_num_embeddings - old_num_embeddings, model_dim, dtype=orig_output_weight.dtype, device=orig_output_weight.device, requires_grad=orig_output_weight.requires_grad), init_std)
                final_state_dict["output.weight"] = torch.cat([
                    orig_output_weight, 
                    new_output_weight
                ], dim=0)


        torch.save(final_state_dict, consolidated_path)
    
    # Convert to DCP format
    print("Converting to DCP format...")
    torch_save_to_dcp(consolidated_path, output_dir)
    print("Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to DCP format"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model name/path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for DCP checkpoint"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model loading",
    )
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Whether to add special tokens to the tokenizer.",
    )

    args = parser.parse_args()
    download_and_convert_hf_to_dcp(args.model, args.output, args.dtype, args.add_special_tokens)