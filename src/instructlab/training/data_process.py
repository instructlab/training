# SPDX-License-Identifier: Apache-2.0

# Standard
from functools import partial
from pathlib import Path
import os

# Third Party
from datasets import load_dataset
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
    eos_tk: list[int],
    max_len: int = 1024,
):
    if len(whole_sentence_tk) >= max_len or len(whole_sentence_tk) < 20:
        return False
    # last token should be eos_token
    if not eos_tk[0] in (
        whole_sentence_tk[-1],
        whole_sentence_tk[-2],
        whole_sentence_tk[-3],
    ):
        return False

    # NOTE - below checks are no longer strictly required, but we may want to revisit to make sure there's nothing we need to bring back in validity checking

    # special_tokens = [system_tk, assistant_tk, user_tk]
    # if not any(token in whole_sentence_tk for token in special_tokens):
    #     return True

    # whole_sentence_tk = np.array(whole_sentence_tk)
    # user_token_index = (whole_sentence_tk == user_tk).nonzero()[0]
    # assistant_token_index = (whole_sentence_tk == assistant_tk).nonzero()[0]
    # eos_token_index = (whole_sentence_tk == eos_tk).nonzero()[0]

    # # check that user_index_token is less than all other indices
    # if (
    #     user_token_index[0] > assistant_token_index[0]
    #     or user_token_index[0] > eos_token_index[0]
    # ):
    #     print("\033[91mthe first sp token is not user_token\033[0m")
    #     log_rank_0(tokenizer.decode(whole_sentence_tk), to_print=True)
    #     return False

    return True


def unmask_message_content(
    example,
    user_tokens,
    assist_tokens,
    system_tokens,
    pretrain_token,
    pretrain_end_token,
    tool_resp_tokens=None,
):
    """
    Create labels for tokens in a sequence with special handling for pretraining tokens and role-specific sequences.

    This function processes a sequence of tokens and generates a corresponding labels list.
    It handles pretraining segments, user/assistant/system role sequences, and ensures proper masking/unmasking
    based on the current context. The function also removes temporary pretraining tokens from the output.

    The labeling follows these rules:
    1. Special token sequences (user, assistant, system) are always masked (-100).
    2. In pretraining segments (between pretrain and pretrain_end tokens), all tokens except special sequences are unmasked.
    3. Outside pretraining segments, only tokens after assistant sequences are unmasked until the next special sequence.
    4. Pretrain and pretrain_end tokens are removed from the final output.

    Parameters:
    - example (dict): A dictionary containing 'input_ids', a list of token IDs.
    - user_tokens (list[int]): The token ID sequence representing the user's turn in the conversation.
    - assist_tokens (list[int]): The token ID sequence representing the assistant's turn in the conversation.
    - system_tokens (list[int]): The token ID sequence representing the system's turn in the conversation.
    - pretrain_token (int): The token ID marking the start of a pretraining segment.
    - pretrain_end_token (int): The token ID marking the end of a pretraining segment.

    Returns:
    - dict: A dictionary with two keys:
        - 'labels': a list of labels for the input tokens, where special sequences and non-assistant responses
                    outside pretraining segments are masked with -100, and all others retain their original token IDs.
        - 'input_ids': a list of the original token IDs with pretraining tokens removed.

    Raises:
    - AssertionError: If any of the following conditions are not met:
        1. Special token sequences are unmasked.
        2. Pretrain tokens are present in the final sentence.
        3. Labels are not aligned with the sentence tokens (when not masked).

    Example:
    >>> example = {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    >>> user_tokens = [2, 3]
    >>> assist_tokens = [5, 6]
    >>> system_tokens = [8, 9]
    >>> pretrain_token = 1
    >>> pretrain_end_token = 10
    >>> result = unmask_message_content(example, user_tokens, assist_tokens, system_tokens, pretrain_token, pretrain_end_token)
    >>> print(result)
    {'labels': [-100, -100, -100, 4, -100, -100, 7, -100, -100], 'input_ids': [2, 3, 4, 5, 6, 7, 8, 9]}

    Note:
    - The function assumes that pretrain and pretrain_end tokens are single integers, not sequences.
    - Special token sequences (user, assistant, system) can be of different lengths.
    - The function handles edge cases such as overlapping sequences and sequences at the start/end of the input.
    """
    sentence_tk = example["input_ids"]
    labels = [-100] * len(sentence_tk)

    def check_sequence(tokens, start_idx):
        return tokens == sentence_tk[start_idx : start_idx + len(tokens)]

    def find_longest_match(start_idx, sequences):
        return max(
            (
                seq
                for seq in sequences
                if seq
                and len(sentence_tk) - start_idx >= len(seq)
                and check_sequence(seq, start_idx)
            ),
            key=len,
            default=None,
        )

    special_sequences = [user_tokens, assist_tokens, system_tokens]
    if tool_resp_tokens:
        special_sequences.append(tool_resp_tokens)

    in_pretraining = False
    unmasking = False
    i = 0
    while i < len(sentence_tk):
        if sentence_tk[i] == pretrain_token:
            in_pretraining = True
            i += 1
            continue
        if sentence_tk[i] == pretrain_end_token:
            in_pretraining = False
            i += 1
            continue

        match = find_longest_match(i, special_sequences)
        if match:
            unmasking = match == assist_tokens
            i += len(match)
            continue

        if in_pretraining or unmasking:
            labels[i] = sentence_tk[i]
        i += 1

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

    # Assertions
    # 1. No special sequence of tokens should be unmasked
    for i in range(len(final_sentence_tk)):
        for seq in special_sequences:
            if final_sentence_tk[i : i + len(seq)] == seq:
                assert all(
                    final_labels[i + j] == -100 for j in range(len(seq))
                ), f"Special sequence {seq} is unmasked"

    # 2. No pretrain tokens should be in the final sentence_tk
    assert all(
        token not in [pretrain_token, pretrain_end_token] for token in final_sentence_tk
    ), "Pretrain tokens found in final sentence"

    # 3. The labels have to be aligned with the sentence_tk unless they are masked
    assert all(
        label in (token, -100) for label, token in zip(final_labels, final_sentence_tk)
    ), "Labels are not aligned with sentence tokens"

    return {"labels": final_labels, "input_ids": final_sentence_tk}


def add_is_pretrain_sample(example, pretrain_tk):
    if pretrain_tk in example["input_ids"]:
        example["is_pretrain"] = True


def print_masked_samples(data, tokenizer, is_pretrain, num_proc):
    def get_masked_and_orig_text(sample):
        labels = sample["labels"]
        input_ids = sample["input_ids"]
        mask_id = get_sp_token(tokenizer, "<|MASK|>")[0]
        label = [mask_id if tk == -100 else tk for tk in labels]
        text = tokenizer.decode(label)
        orig_text = tokenizer.decode(input_ids)
        return text, orig_text

    filtered_data = data.filter(
        lambda x: x["is_pretrain"] == is_pretrain, num_proc=num_proc
    )
    if len(filtered_data) > 0:
        filtered_data = filtered_data.shuffle()
        for i, sample in enumerate(filtered_data):
            text, orig_text = get_masked_and_orig_text(sample)
            print(f"\033[35mOriginal Input: {orig_text}\n\033[0m")
            print(
                f"\033[33m{'Pretraining' if is_pretrain else 'Instruction'} ex sample {i+1}: {text}\033[0m"
            )
            if i > 1:
                break


def main(args: DataProcessArgs):
    if not os.path.exists(args.data_output_path):
        os.makedirs(args.data_output_path, exist_ok=True)
    print("\033[92m data arguments are:\033[0m")
    print("\033[36m" + args.model_dump_json() + "\033[0m")
    NUM_PROC = args.num_cpu_procs
    CHAT_TEMPLATE, SPECIAL_TOKENS = retrieve_chat_template(args.chat_tmpl_path)
    tokenizer = setup_tokenizer(args.model_path, SPECIAL_TOKENS, CHAT_TEMPLATE)

    (
        system_tk,
        user_tk,
        assistant_tk,
        eos_tk,
        pad_tk,
        bos_tk,
        start_role_tk,
        end_role_tk,
        _,
    ) = [
        get_sp_token(tokenizer, getattr(SPECIAL_TOKENS, sp).token)
        for sp in SPECIAL_TOKENS.__annotations__.keys()
    ]
    if start_role_tk and end_role_tk:
        system_tk = (
            start_role_tk
            + tokenizer.encode("system", add_special_tokens=False)
            + end_role_tk
        )
        user_tk = (
            start_role_tk
            + tokenizer.encode("user", add_special_tokens=False)
            + end_role_tk
        )
        assistant_tk = (
            start_role_tk
            + tokenizer.encode("assistant", add_special_tokens=False)
            + end_role_tk
        )
        tool_resp_tk = (
            start_role_tk
            + tokenizer.encode("tool_response", add_special_tokens=False)
            + end_role_tk
        )
    else:
        tool_resp_tk = None
    log_rank_0(
        f"Special tokens: eos: {eos_tk}, pad: {pad_tk}, bos: {bos_tk}, system: {system_tk}, user: {user_tk}, assistant: {assistant_tk}"
    )

    # Adding after tokenizer setup as these are temp tokens, not to be saved
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|pretrain|>", "<|/pretrain|>", "<|MASK|>"]}
    )

    try:
        data = load_dataset("json", data_files=args.data_path, split="train")
    except:
        # pylint: disable=raise-missing-from,broad-exception-raised
        raise Exception(
            "Malformed or missing data, please ensure that your dataset is not empty and correctly formatted"
        )

    if data.num_rows == 0:
        raise ValueError(
            "The provided dataset is empty, please make sure that your dataset contains samples and try again."
        )

    print(f"\033[92mtokenizing the dataset with {args.model_path} tokenizer...\033[0m")
    data_with_input_ids = data.map(
        lambda x: {
            "input_ids": tokenizer.apply_chat_template(x["messages"], tokenize=True)
        },
        num_proc=NUM_PROC,
    )

    print("\033[38;2;255;165;0mten largest length percentiles:")
    lens = np.array(
        data_with_input_ids.map(
            lambda x: {"len": len(x["input_ids"])}, num_proc=NUM_PROC
        )["len"]
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
        raise RuntimeError(
            f"Dataset does not contain any samples containing less than {args.max_seq_len=} tokens.\nPlease consider increasing your `max_seq_len` value, or adding more samples."
        )

    lowest_10_percent = np.quantile(lens, (0 + np.arange(11)) / 100.0)
    for i, q in enumerate(lowest_10_percent):
        print(f"quantile {i}th: {q}")
    num_dropped_samples = np.sum(lens < 20)
    print(
        f"\033[36mat 20 min sequence length, the number of samples to be dropped is {num_dropped_samples}\033[0m"
    )
    # from ipdb import set_trace; set_trace()
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
        num_proc=NUM_PROC,
    )
    log_rank_0(
        f"\033[33mnumber of dropped samples: {len(data) - len(data_with_input_ids)} -- out of {len(data)}\033[0m"
    )

    print("\033[92mCategorizing training data type...\033[0m")
    data_with_input_ids = data_with_input_ids.map(
        lambda x: {
            "is_pretrain": get_sp_token(tokenizer, "<|pretrain|>")[0] in x["input_ids"]
        },
        num_proc=NUM_PROC,
    )

    _prefill_unmask_message_content = partial(
        unmask_message_content,
        user_tokens=user_tk,
        assist_tokens=assistant_tk,
        system_tokens=system_tk,
        pretrain_token=get_sp_token(tokenizer, "<|pretrain|>")[0],
        pretrain_end_token=get_sp_token(tokenizer, "<|/pretrain|>")[0],
        tool_resp_tokens=tool_resp_tk,
    )
    print("\033[92munmasking the appropriate message content...\033[0m")
    data_with_labels = data_with_input_ids.map(
        _prefill_unmask_message_content,
        num_proc=NUM_PROC,
    )

    print("\033[92m Samples Previews...\033[0m")
    print("\033[92m \n \033[0m")
    print_masked_samples(
        data_with_labels,
        tokenizer,
        is_pretrain=True,
        num_proc=NUM_PROC,
    )
    print_masked_samples(
        data_with_labels,
        tokenizer,
        is_pretrain=False,
        num_proc=NUM_PROC,
    )

    # extract only labels and messages formatted into a new dataset
    data_with_labels = data_with_labels.map(
        lambda x: {
            "len": len(x["input_ids"]),
        },
        num_proc=NUM_PROC,
    )
    data_with_labels = data_with_labels.select_columns(["labels", "input_ids", "len"])
    # MASK and both pretrain tokens should not be in the final tokens, those are special tokens added only for data processing purposes.
    max_id = len(tokenizer) - 3
    final_valid_data = data_with_labels.filter(
        lambda x: all(tk < max_id for tk in x["labels"]), num_proc=NUM_PROC
    )
    # Dropping samples that could break training due to oob ids
    if len(final_valid_data) < len(data_with_labels):
        dropped_samples = len(data_with_labels) - len(final_valid_data)
        print(
            f"\033[93mWarning: {dropped_samples} samples were dropped because they contained token IDs greater than or equal to {max_id}.\033[0m"
        )
    # use path to get the stem of the file
    final_valid_data.to_json(Path(args.data_output_path) / "data.jsonl")


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
    parser.add_argument(
        "--num_cpu_procs",
        type=int,
        default=16,
        help="Number of cpu processes for data processing",
    )
    args = parser.parse_args()
    setup_logger(args.logging_level)
    data_process_args = DataProcessArgs(
        data_output_path=args.data_output_path,
        data_path=args.data_path,
        max_seq_len=args.max_seq_len,
        model_path=args.model_name_or_path,
        chat_tmpl_path=args.chat_tmpl_path,
        num_cpu_procs=args.num_cpu_procs,
    )
    main(data_process_args)

"""
python data_process.py --logging_level INFO --data_path "/new_data/refactored/chat-multiturn/oasst2_arena.jsonl" --data_output_path "./" --max_seq_len 4600 --model_name_or_path "mistralai/Mistral-7B-v0.1"
"""
