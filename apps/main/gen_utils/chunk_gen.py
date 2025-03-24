from typing import Callable, List, Any, Tuple, Optional
import numpy as np
import re
import hashlib
import nltk

from apps.main.gen_utils.misc_utils import get_logger
from lingua.tokenizer import SpecialTokens

COMBINED_LATENT_TEXT_KEY = "combined_text_latent"

logger = get_logger(__name__)

def get_text_splitter(split_mode: str):
    """Returns a tuple of (split_function, separator) for different text splitting modes."""

    def paragraph_splitter(text: str) -> List[str]:
        # Split text into paragraphs by double newlines
        # Matches any characters (including newlines) up to a double newline or end of string
        # return re.findall(r'(?:[\s\S]*?(?:\n\s*\n|$))', text)
        return re.findall(r'[^\n\r].*?(?:\n\s*\n|\s*$)', text, re.DOTALL)
    

    def sentence_splitter(text: str) -> List[str]:
        # Get sentence spans from NLTK
        spans = list(nltk.tokenize.punkt.PunktSentenceTokenizer().span_tokenize(text))
        
        # Extract sentences with whitespace by including gap between spans
        result = []
        last_end = 0
        
        for start, end in spans:
            # Include any whitespaces as the prefix of the next unit by starting from the last end
            result.append(text[last_end:end])
            last_end = end
        
        # Add any remaining text after the last sentence
        if last_end < len(text):
            result[-1] = result[-1] + text[last_end:]
        return result

    if split_mode == "word":
        # Match words, with whitespace kept as prefix of the next word
        return lambda text: re.findall(r'\s*\S+', text)
    elif split_mode == "sentence":
        def _split_sentence(text: str) -> List[str]:
            paragraphs = paragraph_splitter(text)
            sentences = []
            for paragraph in paragraphs:
                sentences.extend(sentence_splitter(paragraph))
            return sentences
            
        return _split_sentence
    elif split_mode == "paragraph":
        # Match paragraphs and their trailing newlines/spaces
        return paragraph_splitter
    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

def chunked_generation_wrapper(
    gen_func: Callable,
    units_per_chunk: Optional[int] = None,
    min_num_units_per_chunk: int = 1,
    max_num_units_per_chunk: Optional[int] = None,
    num_chunks: Optional[int] = None,
    min_num_chunks: Optional[int] = None,
    max_num_chunks: Optional[int] = None,
    apply_random_chunking: bool = True,
    chunk_seed: int = 42,
    split_mode: str = "word",
    include_all_prefix_context: bool = False,
    num_prefix_context_units: int = 1,
    chunk_output_key="latent",
    chunk_input_prefix_key = "prefix",
    chunk_input_suffix_key = "suffix",
    chunk_input_key = "chunked_text_pair",
    chunks_key = "chunks",
) -> Callable:
    """This wrapper splits the input samples into multiple chunks and generate latents for each chunk with the provided generation function.

    More specifically, given an input text X_{1:N}, it first randomly sample `num_chunks - 1` indices (i_1, i_2, ..., i_{num_chunks-1}) without replacement as split points. Then group X_{1:N} into `num_chunks` paired (prefix, suffix) chunks: (null, X_{1:i_1}), (X_{1:i_1}, X_{i_1:i_2}), ..., (X_{i_{num_chunks-2}:i_{num_chunks-1}}, X_{i_{num_chunks-1}:N}).
    """
    def chunked_gen_func(samples, shard_idx, input_key="text", output_key=COMBINED_LATENT_TEXT_KEY, **kwargs):
        logger.info(f"Chunking samples with seed {chunk_seed}")
        orig_samples = []

        splitter = get_text_splitter(split_mode)
        separator = ""

        def _chunk_samples_iterator(_samples):
            for sample in _samples:
                text = sample[input_key]
                units = splitter(text)
                total_units = len(units)

                # if num_chunks is None:
                #     assert units_per_chunk is not None, f"units_per_chunk must be provided when num_chunks is not provided"
                #     sample["num_chunks"] = int(np.ceil(total_units / units_per_chunk))
                # else:
                #     sample["num_chunks"] = num_chunks

                # sample["num_chunks"] = min(max(min_num_chunks or 0, sample["num_chunks"]), max_num_chunks or np.inf)
                # sample["num_chunks"] = min(sample["num_chunks"], total_units)

                if units_per_chunk is None:
                    assert num_chunks is not None, f"num_chunks must be provided when units_per_chunk is not provided"
                    avg_units_per_chunk = max(min_num_units_per_chunk or 1, int(total_units / num_chunks))
                else:
                    avg_units_per_chunk = units_per_chunk

                if max_num_chunks is not None or min_num_chunks is not None:
                    logger.warning(f"max_num_chunks and min_num_chunks are deprecated, please use units_per_chunk, max_num_units_per_chunk, and min_num_units_per_chunk instead")

                if apply_random_chunking:
                    # Create a text-specific seed by hashing the text content, that should be invariant across different shard setups
                    # The try logic is probably not necessary, but just in case some edge case happens
                    try:
                        combined_seed = (int(hashlib.md5(text.encode()).hexdigest(), 16) + chunk_seed) & 0xFFFFFFFF
                        text_specific_rng = np.random.default_rng(combined_seed)
                    except (TypeError, ValueError, AttributeError) as e:
                        logger.warning(f"Failed to create text-specific seed for text chunking: {e}. Falling back to chunk_seed.")
                        text_specific_rng = np.random.default_rng(chunk_seed)

                    # chunk_boundaries = sorted(text_specific_rng.choice(range(1, total_units), sample["num_chunks"] - 1, replace=False))
                    # chunk_boundaries = [0] + chunk_boundaries + [total_units]

                    # Sample chunk lengths from normal distribution centered at units_per_chunk
                    remaining_units = total_units
                    chunk_lengths = []

                    assert min_num_units_per_chunk > 0 and min_num_units_per_chunk < avg_units_per_chunk, f"min_num_units_per_chunk must be greater than 0 and less than avg_units_per_chunk"
                    
                    while remaining_units > 0:
                        length = text_specific_rng.poisson(avg_units_per_chunk - min_num_units_per_chunk) + min_num_units_per_chunk
                        length = min(length, max_num_units_per_chunk or remaining_units)
                        length = min(length, remaining_units)
                        chunk_lengths.append(length)
                        remaining_units -= length
                    
                    # Convert lengths to boundaries
                    chunk_boundaries = np.concatenate([[0], np.cumsum(chunk_lengths)])
                    sample["num_chunks"] = len(chunk_lengths)
                else:
                    # apply uniform chunking
                    sample["num_chunks"] = int(np.ceil(total_units / avg_units_per_chunk))
                    chunk_boundaries = np.linspace(0, total_units, sample["num_chunks"] + 1).astype(int)

                orig_samples.append(sample)
                
                for i in range(sample["num_chunks"]):
                    if i == 0:
                        prefix = "N/A"
                    else:
                        if include_all_prefix_context:
                            prefix_start, prefix_end = chunk_boundaries[0], chunk_boundaries[i]
                        else:
                            assert num_prefix_context_units > 0, f"num_prefix_context_units must be greater than 0"
                            prefix_start, prefix_end = chunk_boundaries[max(0, i-num_prefix_context_units)], chunk_boundaries[i]
                        prefix = separator.join(units[prefix_start:prefix_end])

                    suffix_start, suffix_end = chunk_boundaries[i], chunk_boundaries[i + 1]
                    suffix = separator.join(units[suffix_start:suffix_end])
                    
                    chunked_sample = sample.copy()
                    chunked_sample[chunk_input_key] = {
                        chunk_input_prefix_key: prefix,
                        chunk_input_suffix_key: suffix,
                    }
                    yield chunked_sample

        def _combine_outputs(_samples, _outputs):
            sol_token = SpecialTokens.START_OF_LATENT.value
            eol_token = SpecialTokens.END_OF_LATENT.value

            output_idx = 0
            combined_samples = []
            likelihood_keys = [k for k in _outputs[0] if k.startswith("elbo") or k.startswith("loglikelihood")] 
            likelihood_keys.append("num_suffix_token_truncated")  # some suffix tokens may be truncated due to the context length limit, we should sum them up 
            for sample_idx, sample in enumerate(_samples):
                combined_output = ""
                all_chunks = []
                for i in range(sample["num_chunks"]):
                    output = _outputs[output_idx]

                    if chunk_output_key not in output:
                        logger.warning(f"Chunked generation output missing key `{chunk_output_key}` for sample {sample_idx}, chunk {i}. Skipping this chunk output...")
                        latent_text = ""
                    else:
                        latent_text = sol_token + output[chunk_output_key] + eol_token
                    
                    suffix_text = output[chunk_input_key][chunk_input_suffix_key]
                    if i > 0:  # add back the removed separator before the suffix
                        # suffix_text = separator + suffix_text if split_mode == "word" else suffix_text
                        suffix_text = separator + suffix_text
                    combined_output += (latent_text + suffix_text)
                    output_idx += 1

                    chunk_i = {
                        # no need to store prefix, since always the previous suffix
                        chunk_input_suffix_key: suffix_text,
                        chunk_output_key: output.get(chunk_output_key, ""),
                    }

                    for key in likelihood_keys:
                        if key in output:
                            chunk_i[key] = output[key]

                    all_chunks.append(chunk_i)

                sample[output_key] = combined_output
                sample[chunks_key] = all_chunks
                for key in likelihood_keys:
                    if key in all_chunks[0]:
                        sample[f"{key}"] = sum([chunk[key] for chunk in all_chunks])
                
                combined_samples.append(sample)

            assert output_idx == len(_outputs), f"The number of outputs does not match the number of samples ({output_idx} vs {len(_outputs)})"

            return combined_samples
        
        chunked_outputs, chunk_stats = gen_func(_chunk_samples_iterator(samples), shard_idx, input_key=chunk_input_key, output_key=chunk_output_key, **kwargs)
        outputs = _combine_outputs(orig_samples, chunked_outputs)

        stats = {}
        stats["total_num_samples"] = len(outputs)
        stats[f"total_{input_key}_length"] = chunk_stats[f"avg_{chunk_input_key}_length"] * len(chunked_outputs) // (1 + num_prefix_context_units)  # this is an estimate, since we may count all the prefix and suffix units
        stats[f"avg_{input_key}_length"] = stats[f"total_{input_key}_length"] / len(outputs)
        stats[f"total_{chunk_output_key}_length"] = chunk_stats[f"avg_{chunk_output_key}_length"] * len(chunked_outputs)
        stats[f"avg_{chunk_output_key}_length"] = stats[f"total_{chunk_output_key}_length"] / len(outputs)
        stats[f"total_{output_key}_length"] = stats[f"total_{input_key}_length"] + stats[f"total_{chunk_output_key}_length"]
        stats[f"avg_{output_key}_length"] = stats[f"total_{output_key}_length"] / len(outputs)
        
        return outputs, stats

    return chunked_gen_func



if __name__ == "__main__":
    splitter = get_text_splitter("sentence")
    # splitter = get_text_splitter("paragraph")
    
    text = """> For example, the problem of determining whether a graph has a Hamiltonian cycle is in NP; given a path in the graph, it is fast to verify whether or not it is a Hamiltonian cycle. Submitter is a bit out of touch with reality if he thinks "Non-Mathematicians" will get that example. Nice read overall though.

 People who are good at math, but not mathematicians, will get it. There is very little hope of explaining P and NP to people who feel nervous in the presence of math and don't want to hear
what a "polynomial" is.
 Speaking of math, do you think you could throw a few integrals into some post, or a sequel to "Technical Explanation of Technical Explanations"? I have a friend who's mildly interested in LW because I read it, but isn't willing to read enough to solidify an opinion without that s-shaped signal of competence.
 Or a Hamiltonian cycle...
 If you read the entire section from the beginning, you'll notice that you don't really need to know what a Hamiltonian cycle is to
understand the explanation. A made-up property would have worked equally well.
 Actually the concept of Hamiltonian cycles is decidedly easier and less mathy than even basic algebra. At Kansas University the lowest level math class was called Topics, and one of the areas covered was graph theory, including Hamiltonian cycles. I don't recall anyone having problems with the concept, whereas the same group of kids would struggle to add fractions.
 For P/NP examples, I find SAT the easiest to
explain to people not in CS. It's just: there are some things that can be true or false, and some statements about them using and, or, and not. You want to find a set of T/F assignments that makes all those statements true. And it's intuitively obvious to people that it's much easier to check if an assignment is right than to find one.
 Sure, any reasonably intelligent person can grasp what a Hamiltonian Cycle is, once you explain it to them. But this article starts talking about 'em before it's
defined 'em, which is just bad pedagogy.Sure, you don't have to know what it is to understand the rest of the article, but it's still a pretty valid criticism.(I did three years of undergrad maths and I had to go look up what a Hamiltonian cycle is.)

Search:"""
    units = splitter(text)
    print(units)
    print("Num units:", len(units))
    assert "".join(units) == text