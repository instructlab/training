# SPDX-License-Identifier: Apache-2.0

# Standard
from functools import partial
from pathlib import Path
import os
import time
import typing as t

# Third Party
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np
import regex as re

# First Party
from instructlab.training.config import DataProcessArgs
from instructlab.training.tokenizer_utils import get_sp_token, setup_tokenizer
from instructlab.training.type_definitions import Message, ProcessedMessagesData
from instructlab.training.utils import log_rank_0, retrieve_chat_template, setup_logger

# Constants
MASK_TOKEN = "<|MASK|>"
UNMASK_BEGIN_TOKEN = "<|UNMASK_BEGIN|>"
UNMASK_END_TOKEN = "<|UNMASK_END|>"


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
            unmasking = (match == assist_tokens) or (
                example["unmask"] and match != system_tokens
            )
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


def print_masked_samples(data, tokenizer, unmask, num_proc):
    def get_masked_and_orig_text(sample):
        labels = sample["labels"]
        input_ids = sample["input_ids"]
        mask_id = get_sp_token(tokenizer, MASK_TOKEN)[0]
        label = [mask_id if tk == -100 else tk for tk in labels]
        text = tokenizer.decode(label)
        orig_text = tokenizer.decode(input_ids)
        return text, orig_text

    filtered_data = data.filter(
        lambda x: x["unmask"] == unmask,
        num_proc=num_proc,
        desc="Filtering out pretraining samples",
    )
    if len(filtered_data) > 0:
        filtered_data = filtered_data.shuffle()
        for i, sample in enumerate(filtered_data):
            text, orig_text = get_masked_and_orig_text(sample)
            print(f"\033[35mOriginal Input: {orig_text}\n\033[0m")
            print(
                f"\033[33m{'Pretraining' if unmask else 'Instruction'} ex sample {i + 1}: {text}\033[0m"
            )
            if i > 1:
                break


def process_messages_into_input_ids_with_chat_template(args: DataProcessArgs):
    if not os.path.exists(args.data_output_path):
        os.makedirs(args.data_output_path, exist_ok=True)
    print("\033[92m data arguments are:\033[0m")
    print("\033[36m" + args.model_dump_json() + "\033[0m")
    NUM_PROC = args.num_cpu_procs

    _, SPECIAL_TOKENS = retrieve_chat_template(args.chat_tmpl_path)
    tokenizer = setup_tokenizer(args.model_path, args.chat_tmpl_path)

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
            "input_ids": tokenizer.apply_chat_template(x["messages"], tokenize=True),
            "unmask": bool(x["unmask"]) if "unmask" in x else False,
        },
        num_proc=NUM_PROC,
        desc="Tokenizing the dataset",
    )

    print("\033[38;2;255;165;0mten largest length percentiles:")
    data_with_input_ids = data_with_input_ids.map(
        lambda x: {
            "len": len(x["input_ids"]),
        },
        num_proc=NUM_PROC,
        desc="Calculating length of tokenized samples",
    )
    lens = np.array(data_with_input_ids["len"])
    biggest_10_percent = np.quantile(lens, (90 + np.arange(11)) / 100.0)
    for i, q in enumerate(biggest_10_percent):
        print(f"quantile {90 + i * 1}th: {q}")
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
        desc="Checking the validity of the samples",
    )
    log_rank_0(
        f"\033[33mnumber of dropped samples: {len(data) - len(data_with_input_ids)} -- out of {len(data)}\033[0m"
    )

    print("\033[92mCategorizing training data type...\033[0m")
    data_with_input_ids = data_with_input_ids.map(
        lambda x: {
            "unmask": (
                ("unmask" in x and x["unmask"])
                or (get_sp_token(tokenizer, "<|pretrain|>")[0] in x["input_ids"])
            )
        },
        num_proc=NUM_PROC,
        desc="Categorizing training data type",
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
        desc="Unmasking the appropriate message content",
    )

    print("\033[92m Samples Previews...\033[0m")
    print("\033[92m \n \033[0m")
    print_masked_samples(
        data_with_labels,
        tokenizer,
        unmask=True,
        num_proc=NUM_PROC,
    )
    print_masked_samples(
        data_with_labels,
        tokenizer,
        unmask=False,
        num_proc=NUM_PROC,
    )

    data_with_labels = data_with_labels.select_columns(["labels", "input_ids", "len"])
    # MASK and both pretrain tokens should not be in the final tokens, those are special tokens added only for data processing purposes.
    max_id = len(tokenizer) - 3
    final_valid_data = data_with_labels.filter(
        lambda x: all(tk < max_id for tk in x["labels"]),
        num_proc=NUM_PROC,
        desc="Filtering samples down to only those with valid tokens",
    )
    # Dropping samples that could break training due to oob ids
    if len(final_valid_data) < len(data_with_labels):
        dropped_samples = len(data_with_labels) - len(final_valid_data)
        print(
            f"\033[93mWarning: {dropped_samples} samples were dropped because they contained token IDs greater than or equal to {max_id}.\033[0m"
        )
    # use path to get the stem of the file
    final_valid_data.to_json(
        Path(args.data_output_path) / "data.jsonl",
        num_proc=NUM_PROC,
    )


def wrap_masked_messages(
    msgs: t.List[Message], unmask_roles: t.List[str]
) -> t.List[Message]:
    """
    Given a list of messages and a set of roles we want to unmask, return
    a list with the matching messages wrapped with `<|UNMASK_BEGIN|>` and `<|UNMASK_END|>` tokens
    wrapped around the `message.content` field.

    Args:
        msgs (List[Message]): List of messages we want to wrap with unmask tokens.
        unmask_roles (List[str]): The roles whose messages we should wrap.

    Returns:
        List[Message]: The resultant list with all appropriate messages wrapped.
    """
    new_msgs: t.List[Message] = []
    for msg in msgs:
        content = msg["content"]
        if msg["role"] in unmask_roles:
            content = UNMASK_BEGIN_TOKEN + content + UNMASK_END_TOKEN
        new_msgs.append({"role": msg["role"], "content": content})
    return new_msgs


def unmask_messages(
    msgs: t.List[Message],
    tokenizer: PreTrainedTokenizer,
    unmask_roles: t.List[str],
) -> ProcessedMessagesData:
    """
    Algorithm to unmask messages with any arbitrary Tokenizer, provided the following
    conditions are satisfied:

        1. A chat template has been set on the tokenizer
        2. The tokenizer has either an end-of-sequence, or a padding token that can be used as a fallback.


    The algorithm works like this:

        1. Wrap all messages with `<|UNMASK_BEGIN|>` and `<|UNMASK_END|>` special tokens for all roles in `unmask_roles`
        2. Apply the chat template on the resultant messages
        3. Add all IDs seen into input_ids, and only those ids found within the special unmask boundary tokens into labels

    **Note**:
        If a tokenizer has an end-of-sequence token, it is only ever unmasked for the `assistant` role.
        This helps prevent confusion for the model when learning to predict the next token.
        Please reach out to the instructlab/training maintainers if you need this behavior changed.

    Args:
        msgs (List[Message]): A list of messages.
        tokenizer (transformers.PretrainedTokenizer):
            The modified pretrained tokenizer for the model we're generating this data for.
        unmask_roles (List[str]): All of the roles we should unmask messages for.

    Returns:
        Result (ProcessedMessagesData): Dict with the resulting `input_ids`, `labels`, and `len`
    """
    msgs_with_unmasking = wrap_masked_messages(msgs, unmask_roles)
    input_ids = tokenizer.apply_chat_template(msgs_with_unmasking)

    # get the order of unmasked roles
    unmask_roles_order = iter(
        [msg["role"] for msg in msgs_with_unmasking if msg["role"] in unmask_roles]
    )

    # get token ids
    unmask_begin_token_id = tokenizer.encode(
        UNMASK_BEGIN_TOKEN, add_special_tokens=False
    )[0]
    unmask_end_token_id = tokenizer.encode(UNMASK_END_TOKEN, add_special_tokens=False)[
        0
    ]
    eos_token_id = None
    if tokenizer.eos_token is not None:
        eos_token_id = tokenizer.encode(tokenizer.eos_token, add_special_tokens=False)[
            0
        ]

    final_input_ids = []
    final_labels = []
    i = 0
    unmasking = False
    role_being_actively_unmasked = None

    # pylint: disable=too-many-nested-blocks
    while i < len(input_ids):
        tok = input_ids[i]
        if unmasking:
            if tok == unmask_begin_token_id:
                raise ValueError(
                    f'encountered a "{UNMASK_BEGIN_TOKEN}" token while already unmasking. This should never happen, pleas contact the training maintainers.'
                )

            if tok == unmask_end_token_id:
                # we need to just make sure that we unmask the EOS token for the assistant role
                if (
                    eos_token_id is not None
                    and role_being_actively_unmasked == "assistant"
                ):
                    # TODO(osilkin): clean up this portion so that we don't run into race conditions or other bugs
                    i += 1
                    while i < len(input_ids):
                        final_input_ids.append(input_ids[i])
                        final_labels.append(input_ids[i])
                        if input_ids[i] == eos_token_id:
                            break
                        i += 1
                unmasking = False
                role_being_actively_unmasked = None
            else:
                final_input_ids.append(tok)
                final_labels.append(tok)
        else:
            if tok == unmask_end_token_id:
                raise ValueError(
                    f'encountered an "{UNMASK_END_TOKEN}" token while not unmasking. This should never happen, please contact the training maintainers.'
                )

            if tok == unmask_begin_token_id:
                unmasking = True
                role_being_actively_unmasked = next(unmask_roles_order, None)
            else:
                final_input_ids.append(tok)
                final_labels.append(-100)
        i += 1

    # validation logic
    if unmask_begin_token_id in final_input_ids:
        raise ValueError(
            f"{UNMASK_BEGIN_TOKEN} token found in final_input_ids. This should never happen, please contact the training maintainers."
        )
    if unmask_begin_token_id in final_labels:
        raise ValueError(
            f"{UNMASK_BEGIN_TOKEN} token found in final_labels. This should never happen, please contact the training maintainers."
        )
    if unmask_end_token_id in final_input_ids:
        raise ValueError(
            f"{UNMASK_END_TOKEN} token found in final_input_ids. This should never happen, please contact the training maintainers."
        )
    if unmask_end_token_id in final_labels:
        raise ValueError(
            f"{UNMASK_END_TOKEN} token found in final_labels. This should never happen, please contact the training maintainers."
        )

    if role_being_actively_unmasked is not None:
        raise RuntimeError(
            "suffered a critical failure: unmasking finished but not all messages were processed. Please report this!"
        )

    if len(final_input_ids) != len(final_labels):
        raise RuntimeError(
            "suffered a critical failure: final_input_ids and final_labels are not the same length. Please report this!"
        )

    return ProcessedMessagesData(
        input_ids=final_input_ids,
        labels=final_labels,
        len=len(final_input_ids),
    )


def unmask_sample(
    sample: t.Dict[str, t.Any], tokenizer: PreTrainedTokenizer
) -> ProcessedMessagesData:
    """
    Given a sample from a dataset, unmask the appropriate messages and return a sample containing the
    `input_ids` and `labels` fields.

    Args:
        sample: A sample from a dataset.
        tokenizer: The tokenizer to use for unmasking.

    Returns:
        A sample (dict) containing the `input_ids` and `labels` fields.
    """
    # TODO(osilkin): we should define an unmasking policy that
    # enables the user to more dynamically choose what should be unmasked and not.

    # if sample has `unmask` set to true, we unmask everything other than the system role,
    # else we only unmask assistant
    unmask_roles_set = {"assistant"}
    if "unmask" in sample and sample["unmask"]:
        # TODO(osilkin): this computation happens everytime but we could optimize it by getting all
        # the unique roles ahead of time
        unmask_roles_set = set(m["role"] for m in sample["messages"]) - {"system"}

    unmask_roles = list(unmask_roles_set)
    return unmask_messages(sample["messages"], tokenizer, unmask_roles)


def extract_messages_from_pretraining_text(text: str) -> t.List[Message]:
    """
    Given a message from a pretraining message that was formatted using either the generic
    Granite (3.x) template or the legacy Granite 7B template, extract the contents
    and return them as a list of messages.

    Args:
        text (str): The pretraining message to parse.
        use_legacy_pretraining_format (bool):
            Whether or not to parse the message using the legacy format or the generic
            format.
    Returns:
        messages (List[Message]): The list of messages extracted from the sample.
    """
    # Regular expression pattern to match only <|user|> and <|assistant|>
    legacy_pattern = r"<\|(user|assistant)\|>\n([^<]+)"
    generic_pattern = r"<\|start_of_role\|>(user|assistant)<\|end_of_role\|>\n([^<]+)"
    legacy_user_token = "<|user|>"
    legacy_assistant_token = "<|assistant|>"
    legacy_eos_token = "<|endoftext|>"
    generic_eos_token = "<|end_of_text|>"

    use_legacy_template = legacy_user_token in text or legacy_assistant_token in text

    pattern = legacy_pattern if use_legacy_template else generic_pattern
    eot_str = legacy_eos_token if use_legacy_template else generic_eos_token

    extracted_messages: t.List[Message] = []

    # Generator function to process the matches iteratively
    for match in re.finditer(pattern, text):
        role = (match.group(1),)
        content = match.group(2)
        content = content.replace(eot_str, "")
        extracted_messages.append({"role": role[0], "content": content})

    return extracted_messages


def is_pretraining_format(ds: Dataset) -> bool:
    """
    Determine whether or not this is a legacy dataset which needs conversion.
    Legacy == contains "pretraining" roles

    Args:
        ds (Dataset): The dataset to check.

    Returns:
        bool: True if the dataset is legacy, False otherwise.
    """

    # TODO(osilkin): deprecate this eventually
    t1 = time.time()
    print("Checking if the dataset contains any legacy pretraining samples.")
    has_pretrain_roles = "pretraining" in set(ds.flatten()["messages.role"])
    t2 = time.time()
    print(f"Done, took {t2 - t1} seconds")
    return has_pretrain_roles


def pretraining_is_using_legacy_granite_chat_template(ds: Dataset) -> bool:
    """
    Determines whether or not this is using the legacy IBM chat template or the generic chat template.
    I.e.,

    ```
    <|system|>
    You are a friendly AI assistant...
    <|user|>
    Why is the sky blue?
    <|assistant|>
    Great question! What you perceive to be the sky being blue is actually the result of a process known as 'light difraction'...
    <|endoftext|>
    ```

    Args:
        ds (Dataset): The dataset to check.

    Returns:
        bool: True if the dataset is legacy, False otherwise.
    """
    pretraining_msg = None
    for sample in ds:
        if any(m["role"] == "pretraining" for m in sample["messages"]):
            pretraining_msg = [
                m for m in sample["messages"] if m["role"] == "pretraining"
            ][0]
            break

    if not pretraining_msg:
        raise ValueError("could not find any pretraining messages")

    msg_content = pretraining_msg["content"]

    if "<|user|>" in msg_content:
        # quick sanity check to ensure that the special tokens we expect to be in the message are there
        assert "<|user|>" in msg_content and "<|assistant|>" in msg_content
        return True
    else:
        # quick sanity check to ensure that the special tokens we expect to be in the message are there
        assert "<|start_of_role|>user<|end_of_role|>" in msg_content
        assert "<|start_of_role|>assistant<|end_of_role|>" in msg_content
        return False


def ensure_dataset_is_compatible_with_legacy_format(
    sample: t.Dict[str, t.Any],
) -> t.Dict[str, t.Any]:
    """
    Given a sample that uses the legacy pre-training format, we unroll the samples into ones with the
    original messages contents.
    """
    # deepcopy to prevent re-referencing the existing objects
    new_sample = {
        "messages": [],
        "unmask": sample.get("unmask", False),
    }
    for msg in sample["messages"]:
        if msg["role"] != "pretraining":
            new_sample["messages"].append(msg)
            continue

        # handle unmasking
        new_sample["messages"].extend(
            extract_messages_from_pretraining_text(msg["content"])
        )
        new_sample["unmask"] = True

    return new_sample


def filter_samples_by_length(
    data: Dataset,
    max_seq_len: int,
    min_seq_len: int = 20,
    num_proc: int = 1,
) -> Dataset:
    """
    Filter samples by length.

    Args:
        data (Dataset): The dataset to filter.
        max_seq_len (int): The maximum sequence length.
        min_seq_len (int):
            The minimum sequence length -- this is currently set to 20 as the default
            because the old SDG code used to produce broken samples, so this would help filter them out.
            TODO(osilkin): remove this once once we're confident everything is working.
        num_proc (int):
            The number of processes to use for the filtering.
    Returns:
        Dataset: The filtered dataset.
    """
    return data.filter(
        lambda x: min_seq_len <= x["len"] < max_seq_len, num_proc=num_proc
    )


def process_messages_into_input_ids(
    data_path: str,
    data_output_path: str,
    max_seq_len: int,
    model_path: str,
    num_cpu_procs: int,
) -> None:
    """
    Process data for training using the updated unmasking logic.

    This function orchestrates the data processing pipeline by delegating to specialized functions:
    1. Sets up the output directory
    2. Loads and validates the dataset
    3. Configures the tokenizer with special tokens
    4. Processes samples to generate input_ids and labels
    5. Analyzes dataset statistics
    6. Previews samples for verification
    7. Prepares and saves the final dataset

    Args:
        data_path: Path to the input dataset
        data_output_path: Directory in which to save the processed dataset
        max_seq_len: Maximum sequence length for filtering samples
        model_path: Path to the pre-trained model
        num_cpu_procs: Number of CPU processes for parallel processing

    Returns:
        None
    """
    # validate that we can even write to the intended directory before
    # spending potentially a long time processing the dataset only to find out
    # that we can't write to the directory
    ensure_can_write_to_directory(data_output_path)

    data = load_and_validate_dataset(data_path, num_cpu_procs)
    tokenizer = configure_tokenizer(model_path)
    data_with_input_ids_and_labels = process_samples(data, tokenizer, num_cpu_procs)

    # provide an analysis of dataset statistics -- for legacy compatibility
    analyze_dataset_statistics(
        data_with_input_ids_and_labels, max_seq_len, num_cpu_procs
    )

    # filter samples down so they are within the max sequence length
    data_with_input_ids_and_labels = filter_samples_by_length(
        data_with_input_ids_and_labels, max_seq_len, num_proc=num_cpu_procs
    )

    # preview samples -- for legacy compatibility
    preview_samples(data_with_input_ids_and_labels, tokenizer, num_cpu_procs)

    # save the final dataset
    final_dataset = prepare_final_dataset(
        data_with_input_ids_and_labels, tokenizer, num_cpu_procs
    )
    save_dataset(final_dataset, data_output_path, num_cpu_procs)


def ensure_can_write_to_directory(output_dir: str) -> None:
    """
    Ensure that we can write to the output directory.

    Args:
        output_dir: Directory to check
    """
    dir_to_check = os.path.abspath(output_dir)
    while not os.path.exists(dir_to_check) and dir_to_check != "/":
        dir_to_check = os.path.dirname(dir_to_check)

    if not os.access(dir_to_check, os.W_OK):
        raise OSError(
            f"Cannot write to '{output_dir}'. Please ensure that you have write permissions to this directory."
        )


def load_and_validate_dataset(data_path: str, num_procs: int) -> Dataset:
    """
    Load and validate the dataset from the specified path.
    If the dataset is in the legacy format, we automatically convert it to the new format.

    Args:
        data_path (str): The path to the dataset.
        num_procs (int): The number of processes to use for the conversion.

    Returns:
        (datasets.Dataset): The dataset.
    """
    try:
        data = load_dataset("json", data_files=data_path, split="train")
    except Exception as e:
        raise ValueError(
            "Malformed or missing data, please ensure that your dataset is not empty and correctly formatted"
        ) from e

    if data.num_rows == 0:
        raise ValueError(
            "The provided dataset is empty, please make sure that your dataset contains samples and try again."
        )

    return data.map(
        ensure_dataset_is_compatible_with_legacy_format,
        num_proc=num_procs,
        desc="Ensuring dataset is compatible with legacy format.",
    )


def configure_tokenizer(model_path: str) -> PreTrainedTokenizer:
    """Configure the tokenizer with necessary special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if not tokenizer.chat_template:
        raise ValueError(
            "Tokenizer doesn't currently have a chat template. Need to support adding one."
        )

    # Add special tokens for masking
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                UNMASK_BEGIN_TOKEN,
                UNMASK_END_TOKEN,
                MASK_TOKEN,
            ]
        }
    )

    return tokenizer


def process_samples(
    data: Dataset, tokenizer: PreTrainedTokenizer, num_cpu_procs: int
) -> Dataset:
    """Process samples to generate input_ids and labels."""

    # Create a wrapper function for unmask_sample
    process_sample_fn = partial(unmask_sample, tokenizer=tokenizer)

    # Process the dataset
    processed_data = data.map(
        process_sample_fn,
        num_proc=num_cpu_procs,
        desc="Converting samples into input_ids and labels...",
        load_from_cache_file=False,
    )

    return processed_data


def analyze_dataset_statistics(
    data: Dataset, max_seq_len: int, num_cpu_procs: int
) -> None:
    """
    Analyze and print dataset statistics.

    Note:
        This is a function used in the legacy data processing script.
        Future support is not guaranteed.
    """
    # Calculate sequence lengths
    lens = np.array(data["len"])

    # Print largest length percentiles
    print("\033[38;2;255;165;0mten largest length percentiles:")
    biggest_10_percent = np.quantile(lens, (90 + np.arange(11)) / 100.0)
    for i, q in enumerate(biggest_10_percent):
        print(f"quantile {90 + i * 1}th: {q}")
    print("\033[0m")

    # Check for samples exceeding max sequence length
    num_dropped_samples = np.sum(lens > max_seq_len)
    print(
        f"\033[36mat {max_seq_len} max sequence length, the number of samples to be dropped is {num_dropped_samples}\033[0m"
    )
    print(f"\033[36m({((num_dropped_samples / len(lens)) * 100):.2f}% of total)\033[0m")

    if num_dropped_samples == len(data):
        raise RuntimeError(
            f"Dataset does not contain any samples containing less than {args.max_seq_len=} tokens.\n"
            f"Please consider increasing your `max_seq_len` value, or adding more samples."
        )

    # Print smallest length percentiles
    lowest_10_percent = np.quantile(lens, (0 + np.arange(11)) / 100.0)
    for i, q in enumerate(lowest_10_percent):
        print(f"quantile {i}th: {q}")

    # Check for very short samples
    num_dropped_samples = np.sum(lens < 20)
    print(
        f"\033[36mat 20 min sequence length, the number of samples to be dropped is {num_dropped_samples}\033[0m"
    )


def preview_samples(
    data: Dataset, tokenizer: PreTrainedTokenizer, num_proc: int
) -> None:
    """Preview samples from the dataset."""
    print("\033[92m Samples Previews...\033[0m")
    print("\033[92m \n \033[0m")

    # Print pretraining samples
    print_masked_samples(
        data,
        tokenizer,
        unmask=True,
        num_proc=num_proc,
    )

    # Print instruction samples
    print_masked_samples(
        data,
        tokenizer,
        unmask=False,
        num_proc=num_proc,
    )


def prepare_final_dataset(
    data: Dataset, tokenizer: PreTrainedTokenizer, num_proc: int
) -> Dataset:
    """
    Prepare the final dataset for saving.

    Note:
        This is a function used in the legacy data processing script.
        Future support is not guaranteed.
    """
    # drop everything but what's needed for training
    final_data = data.select_columns(["labels", "input_ids", "len"])

    unmask_begin_token_id = tokenizer.encode(
        UNMASK_BEGIN_TOKEN, add_special_tokens=False
    )[0]
    unmask_end_token_id = tokenizer.encode(UNMASK_END_TOKEN, add_special_tokens=False)[
        0
    ]

    def find_samples_with_unmask_ids(sample):
        labels_have_unmask_ids = any(
            # pylint: disable=consider-using-in
            tkid == unmask_begin_token_id or tkid == unmask_end_token_id
            for tkid in sample["labels"]
        )
        input_ids_have_unmask_ids = any(
            # pylint: disable=consider-using-in
            tkid == unmask_begin_token_id or tkid == unmask_end_token_id
            for tkid in sample["input_ids"]
        )
        return labels_have_unmask_ids or input_ids_have_unmask_ids

    invalid_samples = final_data.filter(
        find_samples_with_unmask_ids,
        num_proc=num_proc,
        desc="Validating unmask tokens not in data",
        load_from_cache_file=False,
    )
    if invalid_samples:
        raise ValueError(
            "Unmask tokens found in the processed dataset. This should never happen, please contact the training maintainers."
        )
    return final_data


def save_dataset(dataset: Dataset, output_dir: str, num_proc: int) -> None:
    """
    Save the processed dataset to disk.

    Note:
        This is a function used in the legacy data processing script.
        Future support is not guaranteed.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    dataset.to_json(
        Path(output_dir) / "data.jsonl", num_proc=num_proc, lines=True, orient="records"
    )


def process_data(
    data_path: str,
    data_output_path: str,
    max_seq_len: int,
    model_path: str,
    num_cpu_procs: int,
    chat_tmpl_path: str | None = None,
):
    """
    Process data for training using the updated unmasking logic.
    This serves as the primary entrypoint for data processing script.

    Args:
        data_path (str): Path to the input dataset.
        data_output_path (str): Directory in which to save the processed dataset.
        max_seq_len (int): Maximum sequence length for filtering samples based on the model.
        model_path (str): Path to the pre-trained model or a HF reference.
        num_cpu_procs (int): Number of CPU processes for parallel processing.
        chat_tmpl_path (str):
            Path to the chat template and special tokens. When this argument is used, the legacy data processing method will be used. Otherwise, the new data processing method will be used.
    """
    if chat_tmpl_path:
        print(
            "\033[93mWarning: The legacy data processing method will eventually be deprecated. "
            "Please update your workflow to use the new processing method.\033[0m"
        )
        args = DataProcessArgs(
            data_output_path=data_output_path,
            data_path=data_path,
            max_seq_len=max_seq_len,
            model_path=model_path,
            chat_tmpl_path=chat_tmpl_path,
            num_cpu_procs=num_cpu_procs,
        )
        process_messages_into_input_ids_with_chat_template(args)
    else:
        process_messages_into_input_ids(
            data_output_path=data_output_path,
            data_path=data_path,
            max_seq_len=max_seq_len,
            model_path=model_path,
            num_cpu_procs=num_cpu_procs,
        )


def main(args: DataProcessArgs):
    """
    Process data for training using the updated unmasking logic.
    This serves as the primary entrypoint for data processing script.

    Alias for the `process_data` function.

    Args:
        args (DataProcessArgs): The arguments to pass to the `process_data` function.
    """
    process_data(
        data_path=args.data_path,
        data_output_path=args.data_output_path,
        chat_tmpl_path=args.chat_tmpl_path,
        max_seq_len=args.max_seq_len,
        model_path=args.model_path,
        num_cpu_procs=args.num_cpu_procs,
    )


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
        default=None,
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
    if args.chat_tmpl_path:
        data_process_args = DataProcessArgs(
            data_output_path=args.data_output_path,
            data_path=args.data_path,
            max_seq_len=args.max_seq_len,
            model_path=args.model_name_or_path,
            chat_tmpl_path=args.chat_tmpl_path,
            num_cpu_procs=args.num_cpu_procs,
        )
        process_messages_into_input_ids_with_chat_template(data_process_args)
    else:
        process_messages_into_input_ids(
            data_path=args.data_path,
            data_output_path=args.data_output_path,
            max_seq_len=args.max_seq_len,
            model_path=args.model_name_or_path,
            num_cpu_procs=args.num_cpu_procs,
        )


"""
python data_process.py --logging_level INFO --data_path "/new_data/refactored/chat-multiturn/oasst2_arena.jsonl" --data_output_path "./" --max_seq_len 4600 --model_name_or_path "mistralai/Mistral-7B-v0.1"
"""
