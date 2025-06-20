# Standard
from pathlib import Path
import argparse
import os

# First Party
from instructlab.training import FSDPOptions, TorchrunArgs, TrainingArgs, run_training

# MODIFY THESE VALUES
MAX_BATCH_LENGTH = 10000
MAX_SEQ_LEN = 4096


def run_test(knowledge_data_path, skills_data_path, nnodes, node_rank, nproc_per_node):
    phase1_model_path = os.path.abspath(
        os.path.expanduser("~/.cache/instructlab/models/instructlab/granite-7b-lab")
    )
    phase1_checkpoint_dir = os.path.abspath(
        os.path.expanduser("~/.local/share/instructlab/phased/phase1/checkpoints")
    )
    phase2_checkpoint_dir = os.path.abspath(
        os.path.expanduser("~/.local/share/instructlab/phased/phase2/checkpoints")
    )
    data_output_path = os.path.abspath(
        os.path.expanduser("~/.local/share/instructlab/internal")
    )
    torch_args = TorchrunArgs(
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        node_rank=node_rank,
        rdzv_id=123,
        rdzv_endpoint="0.0.0.0:1738",
    )
    num_phases = 2
    effective_batch_size = 32
    data_path = knowledge_data_path
    ckpt_dir = phase1_checkpoint_dir
    model_path = phase1_model_path
    for phase in range(1, num_phases + 1):
        if phase == 2:
            model_path = os.path.join(phase1_checkpoint_dir, "hf_format")
            checkpoints = list(Path(model_path).iterdir())
            checkpoints = sorted(
                checkpoints,
                reverse=True,
                # sorts based on samples_X
                key=lambda x: int(str(x).rsplit("_", maxsplit=1)[-1]),
            )

            model_path = str(checkpoints[0])
            effective_batch_size = 32
            data_path = skills_data_path
            ckpt_dir = phase2_checkpoint_dir
        training_args = TrainingArgs(
            data_path=data_path,
            model_path=model_path,
            ckpt_output_dir=ckpt_dir,
            data_output_dir=data_output_path,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_len=MAX_BATCH_LENGTH,
            num_epochs=2,
            warmup_steps=10,
            learning_rate=2e-5,
            save_samples=0,
            effective_batch_size=effective_batch_size,
            accelerate_full_state_at_epoch=True,
            checkpoint_at_epoch=True,
            use_liger=False,
            distributed_backend="fsdp",
            process_data=True,
        )
        run_training(torch_args, training_args)
