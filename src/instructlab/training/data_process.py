# Standard
from functools import partial
from pathlib import Path
import os
import random

# Third Party
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np

# First Party
from instructlab.training.config import DataProcessArgs
from instructlab.training.tokenizer_utils import get_sp_token, setup_tokenizer
from instructlab.training.utils import log_rank_0, retrieve_chat_template, setup_logger


def check_valid_sample(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    whole_sentence_tk: list[int],
    system_tk: int,
    assistant_tk: int,
    user_tk: int,
    eos_tk: int,
    max_len: int = 1024,
):
    if len(whole_sentence_tk) >= max_len or len(whole_sentence_tk) < 20:
        return False
    # last token should be eos_token
    if not eos_tk in (whole_sentence_tk[-1], whole_sentence_tk[-2]):
        return False

    special_tokens = [system_tk, assistant_tk, user_tk]
    if not any(token in whole_sentence_tk for token in special_tokens):
        return True

    whole_sentence_tk = np.array(whole_sentence_tk)
    user_token_index = (whole_sentence_tk == user_tk).nonzero()[0]
    assistant_token_index = (whole_sentence_tk == assistant_tk).nonzero()[0]
    eos_token_index = (whole_sentence_tk == eos_tk).nonzero()[0]

    # check that user_index_token is less than all other indices
    if (
        user_token_index[0] > assistant_token_index[0]
        or user_token_index[0] > eos_token_index[0]
    ):
        print("\033[91mthe first sp token is not user_token\033[0m")
        log_rank_0(tokenizer.decode(whole_sentence_tk), to_print=True)
        return False

    return True


def unmask_message_content(
    example, user_token, assist_token, system_token, pretrain_token, pretrain_end_token
):
    """
    Create labels for tokens in a sequence with special handling for pretraining tokens.

    This function processes a sequence of tokens and generates a corresponding labels list.
    Tokens are masked with -100 unless they are part of the assistant's response or within
    a pretraining segment. Pretraining segments are marked by `pretrain_token` and
    `pretrain_end_token`, within which only user, assistant, and system special tokens are masked.
    It also removes the temporary pretraining tokens from the output 'input_ids'.

    Parameters:
    - example (dict): A dictionary containing 'input_ids', a list of token IDs.
    - user_token (int): The token ID representing the user's turn in the conversation.
    - assist_token (int): The token ID representing the assistant's turn in the conversation.
    - system_token (int): The token ID representing the system's turn in the conversation.
    - pretrain_token (int): The token ID marking the start of a pretraining segment.
    - pretrain_end_token (int): The token ID marking the end of a pretraining segment.

    Returns:
    - dict: A dictionary with two keys:
        - 'labels': a list of labels for the input tokens, where non-assistant and non-pretraining
          tokens are masked with -100, and all others retain their original token IDs.
        - 'input_ids': a list of the original token IDs with pretraining tokens removed.
    """
    sentence_tk = example["input_ids"]

    def unmask(token, special_tokens):
        if token in special_tokens:
            return -100
        return token

    def mask(token, *args):
        return -100

    def update_mask_function(token, in_pretraining, mask_function):
        if token == pretrain_token:
            in_pretraining = True
            mask_function = unmask
        elif token == pretrain_end_token:
            in_pretraining = False
            mask_function = mask
        if not in_pretraining:
            if token == assist_token:  # unmasking because of assistant
                mask_function = unmask
            elif token in [
                user_token,
                system_token,
            ]:  # masking because of user or system
                mask_function = mask
        return in_pretraining, mask_function

    labels = [-100] * len(sentence_tk)
    in_pretraining = False
    mask_function = None

    for i, token in enumerate(sentence_tk):
        in_pretraining, mask_function = update_mask_function(
            token, in_pretraining, mask_function
        )
        labels[i] = mask_function(token, [user_token, assist_token, system_token])

    # Find indices of pretrain tokens and remove them from sentence and labels
    pretrain_indices = [
        i
        for i, token in enumerate(sentence_tk)
        if token in [pretrain_token, pretrain_end_token]
    ]
    final_sentence_tk = [
        token for i, token in enumerate(sentence_tk) if i not in pretrain_indices
    ]
    final_labels = [
        label for i, label in enumerate(labels) if i not in pretrain_indices
    ]

    for label, token in zip(final_labels, final_sentence_tk):
        assert label == -100 or token not in [
            user_token,
            assist_token,
            system_token,
        ], f"Token {token} is unmasked, special tokens should not be unmasked."
        assert (
            token not in [pretrain_token, pretrain_end_token]
        ), f"Token {token} is a pretraining token, it should not be in the final sentence."
        assert label in (
            token,
            -100,
        ), f"unless masked, label should be the same as the token."

    return {"labels": final_labels, "input_ids": final_sentence_tk}


def preview_samples(
    tokenizer, pad_tk, pad_str, labels, input_ids, pretrain_indices, instruct_indices
):
    print(
        "\033[33mThe following are some examples of the processed data, with masked tokens (not to be learned) represented with <mask>. The unmasked tokens are the ones the model will learn to predict. Please review these samples to ensure the model is learning to predict expected tokens.\n\033[0m"
    )
    if pretrain_indices:
        sample_indices = random.sample(pretrain_indices, min(len(pretrain_indices), 2))
        for idx in sample_indices:
            label = [pad_tk if tk == -100 else tk for tk in labels[idx]]
            text = tokenizer.decode(label).replace(pad_str, "<mask>")
            orig_text = tokenizer.decode(input_ids[idx])
            print(f"\033[33mPretraining ex sample {idx+1}: {text}\033[0m")
            print(f"\033[35mOriginal Input: {orig_text}\n\033[0m")
    if instruct_indices:
        sample_indices = random.sample(instruct_indices, min(len(instruct_indices), 2))
        for idx in sample_indices:
            label = [pad_tk if tk == -100 else tk for tk in labels[idx]]
            text = tokenizer.decode(label).replace(pad_str, "<mask>")
            orig_text = tokenizer.decode(input_ids[idx])
            print(f"\033[33mInstruction ex sample {idx+1}: {text}\033[0m")
            print(f"\033[35mOriginal Input: {orig_text}\n\033[0m")


def form_data_pools(data, pretrain_tk):
    """
    Sorts data indices into pretraining and instruction pools
    """
    pretrain_indices = []
    instruct_indices = []
    for i in tqdm(range(len(data)), desc="Data type sorting"):
        if pretrain_tk in data[i]:
            pretrain_indices.append(i)
        else:
            instruct_indices.append(i)
    return pretrain_indices, instruct_indices


def main(args: DataProcessArgs):
    CHAT_TEMPLATE, SPECIAL_TOKENS = retrieve_chat_template(args.chat_tmpl_path)
    tokenizer = setup_tokenizer(args.model_path, SPECIAL_TOKENS, CHAT_TEMPLATE)

    eos_tk = get_sp_token(tokenizer, SPECIAL_TOKENS.eos)
    pad_tk = get_sp_token(tokenizer, SPECIAL_TOKENS.pad)
    if SPECIAL_TOKENS.system:
        system_tk = get_sp_token(tokenizer, SPECIAL_TOKENS.system)
    else:
        system_tk = None
    user_tk = get_sp_token(tokenizer, SPECIAL_TOKENS.user)
    assistant_tk = get_sp_token(tokenizer, SPECIAL_TOKENS.assistant)
    log_rank_0(
        f"eos: {eos_tk}, pad: {pad_tk}, system: {system_tk}, user: {user_tk}, assistant: {assistant_tk}"
    )

    # Adding after tokenizer setup as these are temp tokens, not to be saved
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|pretrain|>", "<|/pretrain|>"]}
    )

    data = load_dataset("json", data_files=args.data_path, split="train")
    if data.num_rows == 0:
        raise ValueError("The provided dataset is empty, please make sure that you were able to generate a dataset and try again.")

    print(f"\033[92mtokenizing the dataset with {args.model_path} tokenizer...\033[0m")
    data_with_input_ids = data.map(
        lambda x: {
            "input_ids": tokenizer.apply_chat_template(x["messages"], tokenize=True)
        },
        num_proc=16,
    )

    print("\033[38;2;255;165;0mten largest length percentiles:")
    lens = np.array(
        data_with_input_ids.map(lambda x: {"len": len(x["input_ids"])}, num_proc=16)[
            "len"
        ]
    )
    biggest_10_percent = np.quantile(lens, (90 + np.arange(11)) / 100.0)
    for i, q in enumerate(biggest_10_percent):
        print(f"quantile {90+i*1}th: {q}")
    print("\033[0m")

    num_dropped_samples = np.sum(lens > args.max_seq_len)
    print(
        f"\033[36mat {args.max_seq_len} max sequence length, the number of samples to be dropped is {num_dropped_samples}\033[0m"
    )
    print(f"\033[36m({((num_dropped_samples / len(lens)) * 100):.2f}% of total)\033[0m")
    if num_dropped_samples == len(data):
        raise RuntimeError(f"Dataset does not contain any samples containing less than {args.max_seq_len=} tokens.\nPlease consider increasing your `max_seq_len` value, or adding more samples.")


    lowest_10_percent = np.quantile(lens, (0 + np.arange(11)) / 100.0)
    for i, q in enumerate(lowest_10_percent):
        print(f"quantile {i}th: {q}")
    num_dropped_samples = np.sum(lens < 20)
    print(
        f"\033[36mat 20 min sequence length, the number of samples to be dropped is {num_dropped_samples}\033[0m"
    )

    print("\033[92mchecking the validity of the samples...\033[0m")
    data_with_input_ids = data_with_input_ids.filter(
        lambda x: check_valid_sample(
            tokenizer,
            x["input_ids"],
            system_tk,
            assistant_tk,
            user_tk,
            eos_tk,
            args.max_seq_len,
        ),
        num_proc=16,
    )
    log_rank_0(
        f"\033[33mnumber of dropped samples: {len(data) - len(data_with_input_ids)} -- out of {len(data)}\033[0m"
    )

    print("\033[92mCategorizing training data type...\033[0m")
    pretrain_indices, instruct_indices = form_data_pools(
        data_with_input_ids["input_ids"],
        pretrain_tk=get_sp_token(tokenizer, "<|pretrain|>"),
    )

    _prefill_unmask_message_content = partial(
        unmask_message_content,
        user_token=user_tk,
        assist_token=assistant_tk,
        system_token=system_tk,
        pretrain_token=get_sp_token(tokenizer, "<|pretrain|>"),
        pretrain_end_token=get_sp_token(tokenizer, "<|/pretrain|>"),
    )
    print("\033[92munmasking the appropriate message content...\033[0m")
    data_with_labels = data_with_input_ids.map(
        _prefill_unmask_message_content,
        num_proc=16,
    )

    preview_samples(
        tokenizer,
        pad_tk,
        SPECIAL_TOKENS.pad,
        data_with_labels["labels"],
        data_with_labels["input_ids"],
        pretrain_indices,
        instruct_indices,
    )

    # extract only labels and messages formatted into a new dataset
    data_with_labels = data_with_labels.select_columns(["labels", "input_ids"])
    # use path to get the stem of the file
    data_with_labels.to_json(Path(args.data_output_path) / f"data.jsonl")


if __name__ == "__main__":
    # Standard
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess a dataset for training a language model"
    )
    parser.add_argument(
        "--logging_level", type=str, default="INFO", help="Logging level"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset file"
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        required=True,
        help="Path to the output dataset file",
    )
    parser.add_argument(
        "--max_seq_len", type=int, required=True, help="Maximum sequence length"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="Model name or path"
    )
    parser.add_argument(
        "--chat-tmpl-path",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "chat_templates/ibm_generic_tmpl.py"
        ),
        help="Path to desired chat template and special tokens, defaults to IBM generic.",
    )
    args = parser.parse_args()
    setup_logger(args.logging_level)
    data_process_args = DataProcessArgs(
        data_output_path=args.data_output_path,
        data_path=args.data_path,
        max_seq_len=args.max_seq_len,
        model_path=args.model_name_or_path,
        chat_tmpl_path=args.chat_tmpl_path,
    )
    main(data_process_args)

"""
python data_process.py --logging_level INFO --data_path "/new_data/refactored/chat-multiturn/oasst2_arena.jsonl" --data_output_path "./" --max_seq_len 4600 --model_name_or_path "mistralai/Mistral-7B-v0.1"
"""
