# SPDX-License-Identifier: Apache-2.0

# Standard
from enum import StrEnum
from functools import partial
from pathlib import Path
import os
import typing as t
import regex as re

# Third Party
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
import numpy as np
from tqdm import tqdm

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
            "input_ids": tokenizer.apply_chat_template(x["messages"], tokenize=True),
            "unmask": bool(x["unmask"]) if "unmask" in x else False,
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
            "is_pretrain": (
                get_sp_token(tokenizer, "<|pretrain|>")[0] in x["input_ids"]
            )
            or x["unmask"]
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


class UnmaskPolicy(StrEnum):
    ALL_BUT_SYSTEM = "all-but-system"
    ASSISTANT = "assistant"

# this is what we will use as a placeholder
PLACEHOLDER_GLYPH = ''

def placeholder_msgs(msgs: t.List[t.Dict[str, str]]):
    return [{"role": m["role"], "content": PLACEHOLDER_GLYPH} for m in msgs]

# basically the algorithm will look like this:
# 1. given some list of messages, create a template set of messages with the contents replaced with a glyph
# 2. tokenize the glyph messages and identify the portions in the message where the glyph exists
# 3. with the tokenized list, identify the ranges where the glyph exists. We will want to replace these ranges with tokenized copies of each message
# 4. with the knowledge of where the new message ranges are, we can now unmask according to our policy
#   1. create a copy of the input IDs and leave the portions masked (-100) except for where we expect them to be unmasked
#   2. when unmasking a particular message, if the tokenizer has an EOS token, assert that it is last token 


def get_placeholder_ranges(placeholder_ids: t.List[int], tokenizer: PreTrainedTokenizer):
    glyph_id = tokenizer.encode(PLACEHOLDER_GLYPH, add_special_tokens=False)  # we want to ignore special tokens since we're just extracting the token IDs here
    ranges = []
    i = 0
    while i < len(placeholder_ids):
        # look to start substring matching
        if placeholder_ids[i] == glyph_id[0]:
            j = i
            k = 0
            matching = True
            while k < len(glyph_id) and j < len(placeholder_ids):
                # keep looking to see how far we can match against the glyphd ID
                if placeholder_ids[j] != glyph_id[k]: 
                    matching = False
                    break

                j += 1
                k += 1

            # we were able to loop through successfully
            if k == len(glyph_id) and matching:
                # we now know that between `starti` and `i` there exists a range which is part of a tokenizer
                ranges.append((i, j))

                # now we can set `i` <-- j, and set `starti` <-- j + 1
                i = j
        i += 1

    # assert len(ranges) > 1
    return ranges


def unmask_messages(msgs: t.List[t.Dict[str, str]], tokenizer: PreTrainedTokenizer, glyph_tokenizer: PreTrainedTokenizer, unmask_roles: t.List[str] = None) -> t.Dict[str, t.List[int]]:
    """
    Given a list of messages and an arbitrary tokenizer, returns a dictionary with
    `input_ids` and `labels` containing the correct masking.
    """
    # unmask everything
    if not unmask_roles:
        unmask_roles = list(set(m["role"] for m in msgs))

    # first we need to create the placeholder IDs
    placeholder_ids = tokenizer.apply_chat_template(placeholder_msgs(msgs))
    ranges = get_placeholder_ranges(placeholder_ids, glyph_tokenizer)
    individual_msgs = [tokenizer.encode(m["content"], add_special_tokens=False) for m in msgs]  # no special tokens here since we are looking to inject these into a broader template

    final_input_ids = []
    final_labels = []

    j = 0
    while j < len(placeholder_ids):
        # remove one range
        if not ranges:
            # just append everything else to the end
            final_input_ids.extend(placeholder_ids[j:])
            final_labels.extend([-100] * len(placeholder_ids[j:]))
            break
        
        start_idx, end_idx = ranges[0]
        if j < start_idx:
            # default case, just continue adding into input IDs and labels without doing anything
            final_input_ids.append(placeholder_ids[j])
            final_labels.append(-100)   # mask this out, we dont care about it
            j += 1
            continue
        else:
            # otherwise, we now must insert the tokenized user message. We select it via:
            msg_idx = len(individual_msgs) - len(ranges)  # this should always select the correct message
            msg = individual_msgs[msg_idx]

            # msg will go in no matter what
            final_input_ids.extend(msg)

            # check if we should unmask or not
            should_unmask = msgs[msg_idx]["role"] in unmask_roles
            if should_unmask:
                # now we can append the correct message into the input IDs with the proper masking
                final_labels.extend(msg)
            else:
                final_labels.extend([-100] * len(msg))

            # continue only looking at the next set of ranges
            j = end_idx
            ranges = ranges[1:]

            if tokenizer.eos_token_id is not None:
                suffix_start_j = j
                while j < len(placeholder_ids) and placeholder_ids[j] != tokenizer.eos_token_id:
                    j += 1

                if j >= len(placeholder_ids) or placeholder_ids[j] != tokenizer.eos_token_id:
                    raise RuntimeError('failed to find the trailing EOS token id')

                # by now we know that we are both within range and have found the trailing eos token id
                final_input_ids.extend(placeholder_ids[suffix_start_j:j+1])
                unmasked_eos_sequence = placeholder_ids[suffix_start_j:j+1]
                if should_unmask:
                    final_labels.extend(unmasked_eos_sequence)
                else:
                    final_labels.extend([-100] * len(unmasked_eos_sequence))
                j += 1

    return {
        "input_ids": final_input_ids,
        "labels": final_labels
    }



def unmask_sample(sample: t.Dict[str, t.Any], tokenizer: PreTrainedTokenizer, glyph_tokenizer: PreTrainedTokenizer) -> t.Dict[str, t.Any]:
    # determine unmask policy
    # TODO: make this simpler
    policy = UnmaskPolicy.ALL_BUT_SYSTEM if sample["unmask"] else UnmaskPolicy.ASSISTANT
    
    # select roles to unmask
    unmask_roles = {"assistant"}
    if policy == UnmaskPolicy.ALL_BUT_SYSTEM:
        unmask_roles = set(m["role"] for m in sample["messages"]) - {"system"}
    
    unmask_roles = list(unmask_roles)
    return unmask_messages(sample["messages"], tokenizer, glyph_tokenizer, unmask_roles)

    


def extract_legacy_messages_from_text(text: str) -> t.List[t.Dict]:
    # Regular expression pattern to match only <|user|> and <|assistant|>
    pattern = r"<\|(user|assistant)\|>([^<]+)"

    extracted_messages = []

    # Generator function to process the matches iteratively
    for match in re.finditer(pattern, text):
        role = (match.group(1),)
        content = match.group(2)
        content = content.replace("<|endoftext|>", "")
        extracted_messages.append({"role": role[0], "content": content})

    return extracted_messages


def convert_legacy_pretraining_into_new_template(
    samples: t.List[dict], new_model_path: str
):
    new_tokenizer = AutoTokenizer.from_pretrained(new_model_path)

    # count how many we ignored just for funsies
    new_samples = []
    num_ignored = 0
    for sample in tqdm(
        samples, total=len(samples), desc="Converting legacy format into new format..."
    ):
        old_msgs = sample["messages"]

        # print the old content
        # old_content = old_tokenizer.decode(old_tokenizer.apply_chat_template(old_msgs))
        # print(old_content)
        # print("- " * 81)

        has_pretraining_msgs = any(m["role"] == "pretraining" for m in old_msgs)
        if not has_pretraining_msgs:
            num_ignored += 1
            new_samples.append(sample.copy())
            continue

        idx, pretrain_message = next(
            iter(
                [
                    (i, msg)
                    for i, msg in enumerate(old_msgs)
                    if msg["role"] == "pretraining"
                ]
            ),
            None,
        )

        pretraining_inner_msgs = extract_legacy_messages_from_text(
            pretrain_message["content"]
        )
        unique_roles = {m["role"] for m in pretraining_inner_msgs}
        assert len(pretraining_inner_msgs) >= 2
        assert "user" in unique_roles
        assert "assistant" in unique_roles
        inner_tokenized = new_tokenizer.apply_chat_template(pretraining_inner_msgs)

        # hack to get around the <|end_of_text|> at the end
        inner_tokenized_content = (
            new_tokenizer.decode(inner_tokenized)
            .rstrip()  # just in case \n actually isnt there
            .removesuffix("<|end_of_text|>")
            + "\n"
        )

        new_pretraining_msg = {
            "role": "pretraining",
            "content": inner_tokenized_content,
        }
        new_pretraining_msgs = (
            old_msgs[:idx] + [new_pretraining_msg] + old_msgs[idx + 1 :]
        )
        new_sample = sample.copy()
        new_sample["messages"] = new_pretraining_msgs
        new_samples.append(new_sample)

    print(f"print processed {len(new_samples)} samples, num ignored: {num_ignored}")
    return new_samples


def is_pretraining_format(ds: Dataset) -> bool:
    """
    Determine whether or not this is a legacy dataset which needs conversion.
    Legacy == contains "pretraining" roles
    """
    for sample in ds:
        for msg in sample["messages"]:
            if msg['role'] == 'pretraining':
                return True
    return False

def pretraining_is_using_legacy_chat_template(ds: Dataset) -> bool:
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
    """
    pretraining_msg = None
    for sample in ds:
        if any(m["role"] == "pretraining" for m in sample["messages"]):
            pretraining_msg = [m for m in sample['messages'] if m['role'] == 'pretraining'][0]
            break
    
    if not pretraining_msg:
        raise ValueError('could not find any pretraining messages')

    if '<|user|>' in pretraining_msg:
        # quick sanity check to ensure that the special tokens we expect to be in the message are there
        assert '<|user|>' in pretraining_msg and '<|assistant|>' in pretraining_msg
        return True
    else:
        # quick sanity check to ensure that the special tokens we expect to be in the message are there
        assert '<|start_of_role|>user<|end_of_role|>' in pretraining_msg
        assert '<|start_of_role|>assistant<|end_of_role|>' in pretraining_msg
        return False

def convert_legacy_pretraining_messages(ds: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    For every `pretraining` sample, we unroll it back into the regular messages format and then
    provide it with the unmasking field.
    """



def convert_legacy_dataset(ds: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    Given an existing dataset, converts it into one that uses unmasking
    fields to indicate whether or not we have a pretraining sample.
    """

    # the way that this will happen works like this:
    # There are two possible formats:
    # 1. legacy chat template 
    # 2. generic IBM chat template (granite-3.x series)

    # what we will do is convert these two into the new format of unmasked samples by
    # parsing the existing chat template and parsing it into a new series of messages

    if pretraining_is_using_legacy_chat_template(ds):
        converted_ds = convert_legacy_pretraining_messages(ds, tokenizer)
    else:
        converted_ds = convert_generic_pretraining_messages(ds, tokenizer)




def new_main(args: DataProcessArgs):
    """
    This should behave in the same way as the old process data script, but now we can use the newly updated
    logic for performing the unmasking
    """
    if not os.path.exists(args.data_output_path):
        os.makedirs(args.data_output_path, exist_ok=True)
    print("\033[92m data arguments are:\033[0m")
    print("\033[36m" + args.model_dump_json() + "\033[0m")

    NUM_PROC = args.num_cpu_procs

    # load dataset now 
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

    # check if we're in legacy mode
    if is_pretraining_format(data):
        raise ValueError("legacy pretraining datasets are not supported with the new method")



    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # some tokenizers insert a prefix space when starting from the beginnning, so we create a separate one
    # specifically so we can have an accurate mapping into the tokenized chat template
    glyph_tokenizer = AutoTokenizer.from_pretrained(tokenizer.name_or_path, add_prefix_space=False)

    if not tokenizer.chat_template:
        raise ValueError("tokenizer doesn't currently have a chat template. Need to support adding one.")

    # provide with a token for masking
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|MASK|>"]
    })

    unmask_fn = partial(
        unmask_sample,
        tokenizer=tokenizer,
        glyph_tokenizer=glyph_tokenizer,
    )
    data_with_input_ids = data.map(
        unmask_fn,
        num_proc=1,
    )

    # ensure that there are unmasked fields within the labels
    all_have_unmasked = all(any(tok != -100 for tok in sample) for sample in data_with_input_ids["labels"])
    assert all_have_unmasked

    # --------------------------------------------------------
    # by now we have a Dataset containing the input ids and labels, so we can proceed to the next phase
    # --------------------------------------------------------

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


if __name__ == "__main__":
    # Standard
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess a dataset for training a language model"
    )
    parser.add_argument(
        '--legacy', action='store_true', default=False, help='Whether or not we should use the legacy processing script'
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
    if args.legacy:
        main(data_process_args)
    else:
        new_main(DataProcessArgs(
            data_path=args.data_path,
            data_output_path='/home/oleg/Programming/training/test-output',
            max_seq_len=350,
            model_path='ibm-granite/granite-3.1-8b-instruct',
            num_cpu_procs=16
        ))



"""
python data_process.py --logging_level INFO --data_path "/new_data/refactored/chat-multiturn/oasst2_arena.jsonl" --data_output_path "./" --max_seq_len 4600 --model_name_or_path "mistralai/Mistral-7B-v0.1"
"""
