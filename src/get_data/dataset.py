import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.exceptions import ProjectException
from src.logger import get_logger

logger = get_logger(__name__)

class PreTrainingDataset(Dataset):
    """
    Reads a raw text file (one paragraph per line), tokenizes all lines
    into one continuous stream of token IDs, then splits into blocks of
    max_seq_len + 1 tokens.
    Each sample returns:
        - input_ids: tokens[0 : max_seq_len]
        - labels:    tokens[1 : max_seq_len+1]
    This ensures the model always uses only past tokens to predict the next.
    """
    def __init__(self, file_path, tokenizer, max_seq_len=512):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.examples = [] #store the dict file as a list of dicts with keys "input_ids" and "labels"

        if not os.path.exists(file_path):
            raise ProjectException(f"PreTrainingDataset file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Tokenize all lines and flatten into one list
        all_tokens = []
        for line in lines:
            tokens = tokenizer.encode(line)
            all_tokens.extend(tokens)

        # Create the shifted blocks of length max_seq_len
        total_tokens = len(all_tokens)
        num_blocks = (total_tokens - 1) // max_seq_len #this returns the number of full blocks we can create, ignoring any leftover tokens that don't fit into a full block
        for i in range(num_blocks):
            block = all_tokens[i * max_seq_len : (i + 1) * max_seq_len + 1] #this creates a block of max_seq_len + 1 tokens, where the last token is the next token for prediction
            input_ids = torch.tensor(block[:-1], dtype=torch.long)
            labels = torch.tensor(block[1:], dtype=torch.long)
            self.examples.append({"input_ids": input_ids, "labels": labels})

        logger.info(f"PreTrainingDataset: {len(self.examples)} samples from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Returns a dict with keys 'input_ids' and 'labels' for the sample at index idx."""
        return self.examples[idx]


class QADataset(Dataset):
    """
    loads the  flat JSON (list of dicts) with keys:
    'title', 'context', 'question', 'answer', 'is_impossible' (optional)
    Formats each example as a single text:
        <|question|> Q <|context|> C <|answer|> Answer
    Shifts the whole sequence for autoregressive training:
        input_ids = full_ids[:-1]
        labels    = full_ids[1:]
    Labels for the prompt part are set to -100 so the loss is computed
    only on the answer and end-of-text tokens.
    """
    def __init__(self, json_path, tokenizer, max_seq_len=512,
                 question_token="<|question|>", context_token="<|context|>",
                 answer_token="<|answer|>"):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = []

        # Token IDs for special tokens=>converts the special tokens to their corresponding token IDs in the tokenizer's vocabulary, which will be used for masking and prompt construction.
        self.question_token_id = tokenizer.convert_tokens_to_ids(question_token)
        self.context_token_id = tokenizer.convert_tokens_to_ids(context_token)
        self.answer_token_id = tokenizer.convert_tokens_to_ids(answer_token)

        if not os.path.exists(json_path):
            raise ProjectException(f"QADataset file not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            question = item['question']
            context = item['context']
            answer = item['answer']   # may be "unanswerable"

            # Build the full text (without shift)
            prompt = f"{question_token} {question} {context_token} {context} {answer_token}" #eg "<|question|> What is the capital of France? <|context|> France is a country in Europe. <|answer|>"
            full_text = prompt + " " + answer + tokenizer.eos_token #concatenates the prompt, answer, and end-of-sequence token into a single string that will be tokenized and used for training. The model will learn to generate the answer given the question and context as part of the prompt.

            # Tokenize the whole sequence
            tokenized = tokenizer.encode(full_text)
            prompt_len = len(tokenizer.encode(prompt))

            # Ensure we have at least one target token (the answer + eos)
            if len(tokenized) < prompt_len + 1:
                continue

            # Truncation: keep the last (max_seq_len+1) tokens to allow shift
            if len(tokenized) > max_seq_len + 1:
                tokenized = tokenized[-(max_seq_len + 1):]
                # Recalculate prompt_len after truncation
                try:
                    ans_idx = tokenized.index(self.answer_token_id) + 1
                    prompt_len = ans_idx
                except ValueError:
                    continue

            # Autoregressive shift
            input_ids = torch.tensor(tokenized[:-1], dtype=torch.long)
            labels = torch.tensor(tokenized[1:], dtype=torch.long)

            # Mask prompt positions (first prompt_len-1 tokens after shift)
            mask_len = prompt_len - 1
            if mask_len > 0:
                labels[:mask_len] = -100 #this sets the labels for the prompt tokens to -100, which tells the loss function to ignore these positions when computing the loss. This way, the model is only trained to predict the answer and end-of-sequence tokens, not the prompt itself.

            self.examples.append({
                "input_ids": input_ids,
                "labels": labels,
                "question": question,
                "context": context,
                "answer_raw": answer
            })

        logger.info(f"QADataset: {len(self.examples)} samples from {json_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# to be used as collate_fn in DataLoader for both datasets, with appropriate pad_token_id from tokenizer
# its task is to take a list of samples (dicts) from the dataset and combine them into a single batch,
# padding the input_ids and labels to the same length,
# and creating an attention mask. It also passes through the raw question, context, and answer fields for the QA dataset if present.
def collate_fn(batch, pad_token_id):
    """
    Pads sequences to max length in batch.
    input_ids padded with pad_token_id, labels with -100.
    Attention mask: 1 for real tokens, 0 for padding.
    Passes through raw QA fields if present.
    """
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Fast path if all same length- no padding needed
    lengths = [t.size(0) for t in input_ids]
    if len(set(lengths)) == 1:
        padded_input_ids = torch.stack(input_ids, dim=0)
        padded_labels = torch.stack(labels, dim=0)

    else:
        # Pad to max length in batch
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

    attention_mask = (padded_input_ids != pad_token_id).long()
    result = {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": attention_mask
    }

    # Pass through raw strings for QA evaluation
    if "question" in batch[0]:
        result["questions"] = [item["question"] for item in batch]
        result["contexts"] = [item["context"] for item in batch]
        result["answer_raws"] = [item["answer_raw"] for item in batch]

    return result