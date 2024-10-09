

import torch.distributed
from transformers import LlamaForCausalLM, AutoTokenizer
from torch.optim import AdamW
from datasets import load_dataset
import torch
from torch.optim import AdamW
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy
)
from functools import partial
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig
)
from torch.utils.data import DistributedSampler, DataLoader

torch.cuda.memory._record_memory_history()

import os
import numpy as np
from typing import Callable
from torch.nn import functional as F
import argparse
from peft import LoraConfig, LoraModel
from accelerate import Accelerator, FullyShardedDataParallelPlugin


def make_pad_collate_fn(pad_token_id: int | None) -> Callable:
    rank = int(os.environ['LOCAL_RANK'])

    def pad_collate_fn(batch):
        lens = np.array([len(item["input_ids"]) for item in batch])
        max_len = max(lens)

        input_ids = torch.stack(
            [
                F.pad(
                    item["input_ids"],
                    (max_len - len(item["input_ids"]), 0),
                    mode="constant",
                    value=pad_token_id,
                )
                for item in batch
            ]
        )
        labels = torch.stack(
            [
                F.pad(
                    item["labels"],
                    (max_len - len(item["labels"]), 0),
                    mode="constant",
                    value=-100,
                )
                for item in batch
            ]
        )
        num_loss_counted_tokens = (labels != -100).sum()

        # attention_mask = torch.stack(
        #     [
        #         F.pad(
        #             item["attention_mask"],
        #             (max_len - len(item["attention_mask"]), 0),
        #             mode="constant",
        #             value=0,
        #         )
        #         for item in batch
        #     ]
        # )
        print(
            f"\033[96m total tokens: {max_len * len(batch)} num samples: {len(batch)} num padding tokens: {max_len * len(batch) - lens.sum()} - rank: {rank} "
            f"max len: {max_len} min len: {min(lens)} avg len: {lens.mean()} "
            f"num_loss_counted_tokens: {num_loss_counted_tokens}\033[0m"
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            # "num_loss_counted_tokens": num_loss_counted_tokens,
            # "attention_mask": attention_mask,
        }

    return pad_collate_fn


def main(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device(f'cuda:{local_rank}')

    print('LOADING DATASET')
    data_with_labels = load_dataset("json", data_files="/dev/shm/data.jsonl", split='train')
    # mapped = ds.map(
    #     lambda x: {
    #         "input_ids": tokenizer.apply_chat_template(x['messages'])
    #     },
    # )
    
    # data_with_labels = mapped.map(
    #     lambda x: {
    #         'labels': [tok if i > len(x['input_ids'])/2 else -100 for i, tok in enumerate(x['input_ids']) ]
    #     },
    # )
    data_with_labels.set_format(type='torch', columns=['labels', 'input_ids'])
    train_sampler = DistributedSampler(dataset=data_with_labels, rank=rank, num_replicas=world_size)
    collate_fn = make_pad_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(data_with_labels, num_workers=0, pin_memory=False, shuffle=False, sampler=train_sampler, collate_fn=collate_fn, batch_size=args.batch_size)

    torch.distributed.init_process_group('nccl')
    torch.cuda.set_device(local_rank)



    model = LlamaForCausalLM.from_pretrained(model_name,  torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    if args.lora_r > 0:
        lora_config = LoraConfig(
            r=args.lora_r,
            target_modules=["o_proj"],
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = LoraModel(model, lora_config, "default")
    
    wrap_policy=None
    if args.lora_r > 0:
        print('!!!!!!!!!! USING LORA WRAP POLICY !!!!!!!!!!')
        from peft.utils.other import fsdp_auto_wrap_policy
        wrap_policy = fsdp_auto_wrap_policy(model)
    else:  
        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer
            }
        )

    accelerator = Accelerator(
        fsdp_plugin=FullyShardedDataParallelPlugin(
            auto_wrap_policy=wrap_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_POST,
            cpu_offload=CPUOffload(
                offload_params=False,
            ),
            use_orig_params=False,
            mixed_precision_policy=MixedPrecision(
                buffer_dtype=torch.bfloat16,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16
            ),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            limit_all_gathers=True,
        )
    )

    print(f'accelerator device: {accelerator.device}')
    model = accelerator.prepare(model)
    # )


    optimizer = AdamW(model.parameters(), lr=1e-5)
    optimizer, train_loader = accelerator.prepare(optimizer, train_loader)


    model.train()

    n_epochs = 1
    for n in range(1, n_epochs+1):
        train_sampler.set_epoch(n)
        i = 0
        for batch in train_loader:
            # single batch
            # for k in batch:
            #     batch[k] = batch[k].to(device)
            outputs = model(**batch)
            optimizer.zero_grad()
            print(f'[{i+1:3}/{len(train_loader)}] loss: {outputs.loss}')
            outputs.loss.backward()
            optimizer.step()
            # print(torch.cuda.memory_summary())
            del outputs
            i += 1
            torch.cuda.empty_cache()
   
    print('training complete')
    # if local_rank == 0:
    accelerator.save_model(
        model,
        'accelerate-fsdp-pre-model',
    )



    print('waiting for processes to finish')
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

            
def process_data(max_seq_len: int, model_name_or_path: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    ds = load_dataset("json", data_files="/home/oleg/Programming/training/sample-data/train_all_pruned_SDG.jsonl", split='train')
    mapped = ds.map(
        lambda x: {
            "input_ids": tokenizer.apply_chat_template(x['messages'])
        },
    )
    mapped = mapped.filter(
        lambda x: len(x['input_ids']) < max_seq_len
    )
    
    
    mapped_filtered = mapped.map(
        lambda x: {
            'labels': [tok if i > len(x['input_ids'])/2 else -100 for i, tok in enumerate(x['input_ids']) ]
        },
    )
    
    inputs_labels= mapped_filtered.select_columns(['input_ids', 'labels'])
    inputs_labels.to_json('/dev/shm/data.jsonl', )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_batch_len', type=int, default=100)
    parser.add_argument('--lora_r', type=int, default=0)
    parser.add_argument('--lora_modules', nargs='*', default=None)
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--max_seq_len', type=int, default=1024)
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    # if local_rank == 0:
    #     process_data(args.max_seq_len, args.model_name_or_path)
    main(args)