# Standard
from pathlib import Path
import argparse
import os

# Third Party
from transformers import AutoConfig
import torch

# First Party
from instructlab.training.async_logger import AsyncStructuredLogger
from instructlab.training.config import DistributedBackend, ModelTypes
from instructlab.training.model import Accelerator, Checkpointer, Model, setup_optimizer
from instructlab.training.multipack_sampler import (
    find_packing_max_batch_len_and_grad_accum,
)

# to SDK-ify below:
from instructlab.training.token_dataset import setup_dataloader, setup_dataset
from instructlab.training.tokenizer_utils import setup_tokenizer
from instructlab.training.train import train
from instructlab.training.utils import (
    check_flash_attn_enabled,
    load_latest_full_state,
    setup_logger,
)


def main(args):
    # Third Party
    import yaml

    metric_logger = AsyncStructuredLogger(
        "~/.local/share/instructlab/checkpoints"
        + f"/training_params_and_metrics_global{os.environ['RANK']}.jsonl"
    )
    if os.environ["LOCAL_RANK"] == "0":
        print(f"\033[38;5;120m{yaml.dump(vars(args), sort_keys=False)}\033[0m")
        metric_logger.log_sync({"script_params": vars(args)})

    setup_logger("debug")
    tokenizer = setup_tokenizer(
        "~/.cache/instructlab/models/instructlab/granite-7b-lab",
        os.path.join(os.path.dirname(__file__), "chat_templates/ibm_legacy_tmpl.py"),
    )

    model_conf = AutoConfig.from_pretrained(
        pretrained_model_name_or_path="~/.cache/instructlab/models/instructlab/granite-7b-lab"
    )
    args.model_type = model_conf.model_type

    #### distributed init #####
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group("nccl")
    args.global_rank = torch.distributed.get_rank()
    tensor = torch.ByteTensor([False]).cuda()
    torch.distributed.all_reduce(tensor)
    torch.distributed.barrier()

    flash_enabled = True

    # TODO: would like to replace this with either
    # a) a dataset class
    # b) a dataloader class
    # c) a bit of both
    dataset = setup_dataset(
        args.knowledge_data_path,
        mock=False,
        mock_len=None,
    )

    try:
        packing_max_batch_len, grad_accum = find_packing_max_batch_len_and_grad_accum(
            num_gpus=torch.distributed.get_world_size(),
            avg_sample_len=dataset.get_lengths().mean(),
            effective_batch_size=128,
            max_batch_len_per_gpu=10000,
            is_padding=False,
            dataset=dataset,
            seed=42,
        )
        args.sampler = "multipack"
    except RuntimeError as e:
        if os.environ["LOCAL_RANK"] == "0":
            print(f"\033[38;5;120m{e}\033[0m")

        # fallback to grad accum = 1
        # NOTE: packing max batch len will not be used
        packing_max_batch_len = None
        grad_accum = 1
        args.sampler = "distributed"

    # This model class wraps the various AutoModel classes we support
    # based on model_type, and model_path -> choose auto_model
    lora_config = Model.create_lora_config(
        lora_target_modules=args.lora_target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_r=args.lora_r,
    )
    m = Model(
        model_path="~/.cache/instructlab/models/instructlab/granite-7b-lab",
        output_dir="~/.local/share/instructlab/checkpoints",
        lora_config=lora_config,
        distributed_framework=DistributedBackend(args.distributed_training_framework),
        tokenizer=tokenizer,
        model_type=ModelTypes("Liger"),
        flash_enabled=flash_enabled,
        noise_alpha=None,
    )

    args.samples_per_gpu = 128 // grad_accum // torch.distributed.get_world_size()

    train_loader = setup_dataloader(
        dataset,
        tokenizer.pad_token_id,
        num_workers=8,
        use_dolomite=False,
        flash_enabled=flash_enabled,
        max_batch_len=10000,
        packing_max_batch_len=packing_max_batch_len,
        samples_per_gpu=args.samples_per_gpu,
        sampler=args.sampler,
        seed=42,
    )
    if len(train_loader) == 0:
        # this happens sometimes when we have more GPUs than data to process. In this case
        # we should either alert the user to switch samplers, or do it automatically and
        # warn them about it happening
        print(
            "\033[93mThe dataset is too small for multipack to distribute all of the samples across GPUs. Falling back to the distributed sampler!\033[0m"
        )
        train_loader = setup_dataloader(
            dataset,
            tokenizer.pad_token_id,
            num_workers=8,
            use_dolomite=False,
            flash_enabled=flash_enabled,
            max_batch_len=10000,
            packing_max_batch_len=packing_max_batch_len,
            samples_per_gpu=args.samples_per_gpu,
            sampler=args.sampler,
            seed=42,
        )

    if args.local_rank == 0:
        metric_logger.log_sync(
            {
                "num_gpus": torch.distributed.get_world_size(),
                "avg_sample_len": dataset.get_lengths().mean(),
                "effective_batch_size": 128,
                "max_batch_len_per_gpu": 10000,
                "packing_max_batch_len": packing_max_batch_len,
                "grad_accum": grad_accum,
                "num_batches": len(train_loader),
                "avg_samples_per_batch": len(dataset) / len(train_loader),
                "samples_per_gpu": args.samples_per_gpu,
                "total_samples": len(dataset),  # emit the total number of samples
            }
        )
    # accelerator does not need optimizer to init, in fact, the optimizer needs to be initialized AFTER the Accelerator
    accelerator = Accelerator(
        model=m,
        samples_per_gpu=args.samples_per_gpu,
        grad_accum=grad_accum,
        train_loader=train_loader,
        distributed_framework=DistributedBackend.FSDP,
        fsdp_sharding_strategy="SHARD_GRAD_OP",
        fsdp_cpu_offload_params=True,
        save_samples=True,
    )
    # optimizer needs model that has been prepared by accelerator
    # and then accelerator needs to be prepared AGAIN once optimizer is initialized
    optimizer = setup_optimizer(
        model=m,
        cpu_offload=True,
        name=None,  # choose based on backend
        learning_rate=2e-6,
    )
    accelerator.prepare_with_optimizer(
        optimizer=optimizer,
        lr_scheduler="cosine",
        num_epochs=8,
        num_warmup_steps=10,
    )

    checkpointer = Checkpointer(strategy="full_state")
    checkpointer.load_latest_full_state(
        output_dir="~/.local/share/instructlab/checkpoints"
    )
    train(
        model=m,
        optimizer=optimizer,
        accelerator=accelerator,
        checkpointer=checkpointer,
        sampler=args.sampler,
        use_dolomite=False,
        metric_logger=metric_logger,
        output_dir="~/.local/share/instructlab/checkpoints",
        checkpoint_at_epoch=True,
        effective_batch_size=128,
        last_step=1,
        num_epochs=8,
        save_last=True,
    )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge-data-path", type=str)
    args = parser.parse_args()
    main(args)
