from transformers import GPT2TokenizerFast
from src.exceptions import ProjectException
from src.logger import get_logger

logger = get_logger(__name__)

def load_tokenizer(config: dict) -> GPT2TokenizerFast:
    """
    loads the GPT‑2 tokenizer, add special tokens from config, and return it.
    Special tokens added:
      - pad_token (e.g., <|pad|>)
      - additional_special_tokens for Q&A: question, context, answer tokens
    """
    tokenizer_type = config["tokenizer"]["type"]  # "gpt2"
    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_type)
    except Exception as e:
        raise ProjectException(f"Failed to load tokenizer '{tokenizer_type}'", e)

    # adding the pad token
    pad_token = config["tokenizer"]["pad_token"]
    tokenizer.add_special_tokens({"pad_token": pad_token})

    # adding the Q&A special tokens
    special_tokens_list = [
        config["tokenizer"]["special_tokens"]["question"],
        config["tokenizer"]["special_tokens"]["context"],
        config["tokenizer"]["special_tokens"]["answer"]
    ]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})

    # suppress length warnings by setting a large model_max_length -> 
    # this is important for our long-context pretraining and QA tasks, 
    # where we may exceed the default GPT-2 max length of 1024 tokens. 
    # By setting it to a very large number, we avoid warnings about truncation during tokenization.
    tokenizer.model_max_length = 10**9

    logger.info(f"Tokenizer loaded: {tokenizer_type}, vocab size: {len(tokenizer)}")
    return tokenizer

def encode_text(tokenizer, text: str) -> list:
    """this encodes a string to a list of token IDs."""
    return tokenizer.encode(text)

def decode_text(tokenizer, ids: list) -> str:
    """this decodes a list of token IDs to a string."""
    return tokenizer.decode(ids)

#so in a script the tokenizer will loaded as
#tokenizer = load_tokenizer(config)
#to emcode text: input_ids = encode_text(tokenizer, text)
#to decode: text = decode_text(tokenizer, input_ids)