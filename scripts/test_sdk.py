# Standard
from pathlib import Path
import argparse
import datetime
import os
import subprocess

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
from instructlab.training.utils import StreamablePopen, set_random_seed, setup_logger
import instructlab.training.data_process as dp


def main(args):
    """
    This script uses the classes defined in src/instructlab/training and follows a similar flow as main in main_ds.py
    This is separate to ensure our testing uses a consistent script that will catch breakages to the SDK classes
    main and run_training expect a set of arguments specific to ilab, this script only requires the following arguments:

    model_path
    knowledge_data_path
    skills_data_path
    effective_batch_size
    ckpt_dir

    The other arguments are set per phase by the script and are not configurable

    PHASE 1

    EBS = 128
    model = granite-7b-lab
    MBL = 10000
    model_type = Liger
    seed = 42
    nproc_per_node = 4
    nnodes = 1
    use_dolomite = False
    is_padding_free = True
    lr_scheduler = cosine
    num_epochs = 1
    data_path = KNOWLEDGE DATA

    PHASE 2

    EBS = 3840
    model = last CKPT of PHASE 1
    MBL = 10000
    model_type = Liger
    seed = 42
    nproc_per_node = 4
    nnodes = 1
    use_dolomite = False
    is_padding_free = True
    lr_scheduler = cosine
    num_epochs = 1
    data_path = SKILLS DATA

    """

    # Third Party
    import yaml

    # granite teacher model
    # model_path = os.path.abspath(
    #     os.path.expanduser("~/.cache/instructlab/models/instructlab/granite-7b-lab")
    # )
    # data path to put processed data into
    data_output_path = os.path.abspath(
        os.path.expanduser("~/.local/share/instructlab/internal")
    )
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)
    # # checkpoint dir to put ilab checkpoints into.
    # ckpt_dir = os.path.abspath(
    #     os.path.expanduser("~/.local/share/instructlab/checkpoints")
    # )

    data_path = os.path.join(data_output_path, "data.jsonl")
    metric_logger = AsyncStructuredLogger(
        args.ckpt_dir + f"/training_params_and_metrics_global{os.environ['RANK']}.jsonl"
    )
    if os.environ["LOCAL_RANK"] == "0":
        print(f"\033[38;5;120m{yaml.dump(vars(args), sort_keys=False)}\033[0m")
        metric_logger.log_sync({"script_params": vars(args)})

    setup_logger("INFO")
    tokenizer = setup_tokenizer(args.model_path)

    model_conf = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.model_path
    )
    args.model_type = model_conf.model_type

    #### distributed init #####
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.distributed.init_process_group("nccl")
    args.global_rank = torch.distributed.get_rank()
    tensor = torch.ByteTensor([False]).cuda()
    torch.distributed.all_reduce(tensor)
    torch.distributed.barrier()

    flash_enabled = True

    dataset = setup_dataset(
        data_path,
        mock=False,
        mock_len=None,
    )

    try:
        packing_max_batch_len, grad_accum = find_packing_max_batch_len_and_grad_accum(
            num_gpus=torch.distributed.get_world_size(),
            avg_sample_len=dataset.get_lengths().mean(),
            effective_batch_size=args.effective_batch_size,
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
    lora_config = None
    m = Model(
        model_path=args.model_path,
        output_dir=args.ckpt_dir,
        lora_config=lora_config,
        distributed_framework=DistributedBackend.FSDP,
        tokenizer=tokenizer,
        model_type=ModelTypes("Causallm"),
        flash_enabled=flash_enabled,
        noise_alpha=None,
    )

    args.samples_per_gpu = (
        args.effective_batch_size // grad_accum // torch.distributed.get_world_size()
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
                "effective_batch_size": args.effective_batch_size,
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
        fsdp_cpu_offload_params=False,
        save_samples=0,
    )
    # optimizer needs model that has been prepared by accelerator
    # and then accelerator needs to be prepared AGAIN once optimizer is initialized
    optimizer = setup_optimizer(
        model=m,
        cpu_offload=False,
        name=None,  # choose based on backend
        learning_rate=2e-6,
    )
    accelerator.prepare_with_optimizer(
        optimizer=optimizer,
        lr_scheduler="cosine",
        num_epochs=2,
        num_warmup_steps=10,
    )
    # TODO: make this work more seamlessly
    optimizer = accelerator.optimizer
    m = accelerator.model

    checkpointer = Checkpointer(
        strategy="all", model=m, optimizer=optimizer, accelerator=accelerator
    )
    checkpointer.load_latest_full_state(output_dir=Path(args.ckpt_dir))
    train(
        model=m,
        optimizer=optimizer,
        accelerator=accelerator,
        checkpointer=checkpointer,
        sampler=args.sampler,
        use_dolomite=False,
        metric_logger=metric_logger,
        output_dir=args.ckpt_dir,
        checkpoint_at_epoch=True,
        effective_batch_size=args.effective_batch_size,
        last_step=0,
        num_epochs=2,
        save_last=True,
    )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def run_test(knowledge_data_path, skills_data_path, nnodes, node_rank, nproc_per_node):
    phase1_model_path = os.path.abspath(
        os.path.expanduser("~/.cache/instructlab/models/instructlab/granite-7b-lab")
    )
    data_output_path = os.path.abspath(
        os.path.expanduser("~/.local/share/instructlab/internal")
    )

    phase1_checkpoint_dir = os.path.abspath(
        os.path.expanduser("~/.local/share/instructlab/phased/phase1/checkpoints")
    )
    phase2_checkpoint_dir = os.path.abspath(
        os.path.expanduser("~/.local/share/instructlab/phased/phase2/checkpoints")
    )
    num_phases = 2
    effective_batch_size = 32
    data_path = knowledge_data_path
    ckpt_dir = phase1_checkpoint_dir
    model_path = phase1_model_path
    for phase in range(1, num_phases + 1):
        # override model
        # override checkpoints dir
        # override EBS
        if phase == 2:
            model_path = os.path.join(phase1_checkpoint_dir, "hf_format", "last_epoch")
            effective_batch_size = 32
            data_path = skills_data_path
            ckpt_dir = phase2_checkpoint_dir
        print(f"RUNNING PHASE {phase} of {num_phases}")

        dp.process_data(
            data_output_path=data_output_path,
            model_path=model_path,
            data_path=data_path,
            max_seq_len=4096,
            num_cpu_procs=16,
        )

        command = [
            "torchrun",
            f"--nnodes={nnodes}",
            f"--node_rank={node_rank}",
            f"--nproc_per_node={nproc_per_node}",
            f"--rdzv_id=123",
            f"--rdzv_endpoint=127.0.0.1:12222",
            __file__,
            f"--effective-batch-size={effective_batch_size}",
            f"--model-path={model_path}",
            f"--ckpt-dir={ckpt_dir}",
        ]
        process = None
        interrupt: KeyboardInterrupt | Exception | None = None
        failure = False
        try:
            log_path = os.path.abspath(
                os.path.expanduser(
                    f"~/.local/share/instructlab/checkpoints/full_logs_global{node_rank}.log"
                )
            )
            if not os.path.exists(os.path.dirname(log_path)):
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
            process = StreamablePopen(
                log_path,
                command,
            )
            process.listen()
        except KeyboardInterrupt as e:
            print("Training subprocess interrupted by user.")
            interrupt = e
        except Exception as e:
            print("Unexpected exception received during distributed training")
            interrupt = e
        finally:
            if "process" not in locals() or process is None:
                return

            failure = process.poll() != 0
            if not failure:
                print("\033[92mOperation completed successfully! 🎉\033[0m")
            else:
                print(
                    "\033[91mTraining subprocess has not exited yet. Sending SIGTERM.\033[0m"
                )

            process.terminate()
            try:
                print("Waiting for process to exit, 60s...")
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print(
                    "\033[91mTraining subprocess did not terminate before timeout, sending SIGKILL.\033[0m"
                )
                process.kill()

            if interrupt:
                raise interrupt
            if failure:
                raise RuntimeError(
                    "Suffered a failure during distributed training. Please see the training logs for more context."
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--ckpt-dir", type=str)
    args = parser.parse_args()
    set_random_seed(42)
    main(args)
