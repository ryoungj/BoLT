from .dummy import sample_only
from .openai_batch import openai_batch_generation
from .gen_sync import generate_sync
from .chunk_gen import chunked_generation_wrapper

ALL_GENERATION_FUNC = {
    "synthetic": openai_batch_generation,
    "synthetic_sync": generate_sync,
    "sample_only": sample_only,
}

ALL_GENERATION_METHOD = list(ALL_GENERATION_FUNC.keys())