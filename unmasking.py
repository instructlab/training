print('importing necessary libraries')
from transformers import PreTrainedTokenizer, AutoTokenizer
import typing as t

# test cases
tests = [
    ('granite', AutoTokenizer.from_pretrained('ibm-granite/granite-3.1-8b-instruct')),
    ('llama', AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8b-instruct')),
    ('phi', AutoTokenizer.from_pretrained('microsoft/phi-4')),
]

msgs = [
    {
        "content": "Hello world!",
        "role": "system"
    },
    {
        "content": "Holey shit",
        "role": "user"
    },
    {
        "content": "Hello from the other side",
        "role": "assistant"
    }
]


# this is what we will use as a placeholder
GLYPH = "𓀴"

def placeholder_msgs(msgs: t.List[t.Dict[str, str]]):
    return [{"role": m["role"], "content": GLYPH} for m in msgs]

# basically the algorithm will look like this:
# 1. given some list of messages, create a template set of messages with the contents replaced with a glyph
# 2. tokenize the glyph messages and identify the portions in the message where the glyph exists
# 3. with the tokenized list, identify the ranges where the glyph exists. We will want to replace these ranges with tokenized copies of each message
# 4. with the knowledge of where the new message ranges are, we can now unmask according to our policy
#   1. create a copy of the input IDs and leave the portions masked (-100) except for where we expect them to be unmasked
#   2. when unmasking a particular message, if the tokenizer has an EOS token, assert that it is last token 


def get_placeholder_ranges(placeholder_ids: t.List[int], tokenizer: PreTrainedTokenizer):
    glyph_id = tokenizer.encode(GLYPH, add_special_tokens=False)  # we want to ignore special tokens since we're just extracting the token IDs here
    ranges = []
    i = 0
    while i < len(placeholder_ids):
        # look to start substring matching
        if placeholder_ids[i] == glyph_id[0]:
            print(f'potentially found a glyph id match starting at {i=}')
            j = i
            k = 0
            matching = True
            while k < len(glyph_id) and j < len(placeholder_ids):
                # keep looking to see how far we can match against the glyphd ID
                if placeholder_ids[j] != glyph_id[k]: 
                    print(f'but unfortunately, found that at {k=}, {j=}, {placeholder_ids[j]=} != {glyph_id[k]=}')
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

    return ranges



def unmask_messages(msgs: t.List[t.Dict[str, str]], tokenizer: PreTrainedTokenizer, unmask_roles: t.List[str] = None) -> t.Dict[str, t.List[int]]:
    """
    Given a list of messages and an arbitrary tokenizer, returns a dictionary with
    `input_ids` and `labels` containing the correct masking.
    """
    unmask_roles = list(set(m["role"] for m in msgs)) if not unmask_roles else unmask_roles

    # first we need to create the placeholder IDs
    placeholder_ids = tokenizer.apply_chat_template(placeholder_msgs(msgs))
    ranges = get_placeholder_ranges(placeholder_ids, tokenizer)
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

            # we want to also unmask the EOS token if it is present
            print(f"after extending, {j=} is set to {placeholder_ids[j]=}")

            if tokenizer.eos_token_id is not None:
                print('detected eos token id, proceeding forward')
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

for name, tokenizer in tests:
    GLYPH = "𓀴"
    results = unmask_messages(msgs, tokenizer, ["assistant"])
    print('-' * 120)
    print(f"{results['input_ids']=}")
    print(f"{results['labels']=}")

    print('---')
    print("decoded:")
    print(f"input_ids: {tokenizer.decode(results['input_ids'])}")


    # get the groups by splitting between -100
    groups = []
    i = 0
    z = 0
    y = 0
    while i < len(results["labels"]):
        y += 1
        if y >= 10_000:
            print('stuck in outer loop')
            break

        if results["labels"][i] == -100:
            i += 1 
            continue

        # K, now we know the current one is not to be masked
        group = []
        while i < len(results["labels"]) and results["labels"][i] != -100:
            z += 1
            if z >= 10_000:
                print('stuck in inner loop')
                break
            group.append(results["labels"][i])
            i += 1

        groups.append(group)
        i += 1

    for i, group in enumerate(groups):
        print(f"{i} --> {tokenizer.decode(group)}")

    # lets print out all unmasked tokens
    # unmasked = [tok for tok in results["labels"] if tok != -100]
    # print(llama.decode(unmasked))


