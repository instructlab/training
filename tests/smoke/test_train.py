# Standard
from typing import Generator
import os
import pathlib
import shutil
import sys
import tempfile

# Third Party
from datasets import load_dataset
from transformers import AutoModelForCausalLM
import huggingface_hub
import pytest

# First Party
from instructlab.training import data_process
from instructlab.training.config import (
    DataProcessArgs,
    DistributedBackend,
    LoraOptions,
    TorchrunArgs,
    TrainingArgs,
)
from instructlab.training.main_ds import run_training

MINIMAL_TRAINING_ARGS = {
    "max_seq_len": 140,  # this config fits nicely on 4xL40s and may need modification for other setups
    "max_batch_len": 5000,
    "num_epochs": 1,
    "effective_batch_size": 128,
    "save_samples": 0,
    "learning_rate": 1e-4,
    "warmup_steps": 1,
    "random_seed": 43,
    "use_dolomite": False,
    "is_padding_free": False,
    "checkpoint_at_epoch": True,
    "accelerate_full_state_at_epoch": True,
    "process_data": False,  # expect that incoming data has already been prepared and cached.
    "disable_flash_attn": False,
}

DEFAULT_TORCHRUN_ARGS = {
    "nproc_per_node": 4,  # TODO: this is runner-specific. Should parametrize from environment.
    "nnodes": 1,
    "node_rank": 0,
    "rdzv_id": 123,
    "rdzv_endpoint": "127.0.0.1:12345",
}

REFERENCE_TEST_MODEL = "ibm-granite/granite-3.3-2b-instruct"
RUNNER_CPUS_EXPECTED = 4

# Number of samples to randomly sample from the processed dataset for faster training
NUM_SAMPLES_TO_KEEP = 2500


@pytest.fixture(scope="module")
def custom_tmp_dir() -> Generator[pathlib.Path, None, None]:
    """A custom fixture for a temporary directory.
    By default, `tmp_dir` builtin fixture is function-scoped
    but we can reuse the same cached storage between many tests.

    Yields:
        Generator[pathlib.Path, None, None]: path to root directory of temp storage.
    """
    temp_dir = tempfile.mkdtemp()

    temp_path = pathlib.Path(temp_dir)

    yield temp_path

    shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def checkpoint_dir(
    custom_tmp_dir: pathlib.Path,
) -> Generator[pathlib.Path, None, None]:
    """
    Creates a 'checkpoints' directory.
    This directory must be function-scoped because each test
    will create its own checkpoints.
    """
    ckpt_dir = custom_tmp_dir / "checkpoints"
    ckpt_dir.mkdir()

    yield ckpt_dir

    shutil.rmtree(ckpt_dir)


@pytest.fixture(scope="module")
def prepared_data_dir(custom_tmp_dir: pathlib.Path) -> pathlib.Path:
    """Sets up module-scoped temporary dir for storage of preprocessed data.

    Args:
        custom_tmp_dir (pathlib.Path): root dir of temporary storage

    Returns:
        pathlib.Path: path to directory where preprocessed data can be cached
    """
    data_file_dir = custom_tmp_dir / "prepared_data"
    data_file_dir.mkdir()

    return data_file_dir


@pytest.fixture(scope="module")
def cached_model_dir(custom_tmp_dir: pathlib.Path) -> pathlib.Path:
    """Sets up module-scoped temporary dir for storage of model checkpoint

    Args:
        custom_tmp_dir (pathlib.Path): root dir of temporary storage

    Returns:
        pathlib.Path: path to directory where model checkpoint can be cached
    """
    model_dir = custom_tmp_dir / "model"
    model_dir.mkdir()
    return model_dir


@pytest.fixture(scope="module")
def cached_test_model(cached_model_dir: pathlib.Path) -> pathlib.Path:
    """
    Downloads test model artifacts to temporary cache from HF repo.
    Assumes that the artifacts for the tokenizer are in the same repo.

    Some interesting behavior:
    (1) if model is already cached in $HF_HOME/hub/<model> the parameter blobs
        will be copied into the specified `local_dir`. If some remote
        files (like paper.pdf or tokenizer.config) aren't in the HF_HOME
        cache, they'll be pulled and stored in the `local_dir` cache.
    (2) if model is NOT already cached in $HF_HOME/hub/<model>, a reference will
        still be created to it but the downloaded artifacts will not be copied
        back to the HF_HOME cache from the `local_dir`.
    """

    huggingface_hub.snapshot_download(
        repo_id=REFERENCE_TEST_MODEL,
        local_dir=cached_model_dir,
    )

    return cached_model_dir


def this_file_path() -> pathlib.Path:
    """returns the fully qualified path to this file."""
    return pathlib.Path(__file__).resolve()


def repo_root_dir() -> pathlib.Path:
    """returns the fully qualified path to the root of the repo."""
    current_file_path = this_file_path()
    return current_file_path.parents[2]


def data_in_repo_path() -> pathlib.Path:
    """The data that we'll use in these tests is stored in the repo as an artifact.
    This returns a path to the `data.jsonl` file based on this file's location
    in the repo.

    Returns:
        pathlib.Path: Path to a `.jsonl` file for tests
    """
    repo_root = repo_root_dir()
    data_in_repo_path = repo_root / "sample-data" / "train_all_pruned_SDG.jsonl"
    return data_in_repo_path


def chat_template_in_repo_path() -> pathlib.Path:
    """The chat template that we'll use in these tests is stored in the repo as an artifact.
    This returns a path to the `chattemplate.py` file based on this file's location
    in the repo.

    Returns:
        pathlib.Path: Path to a `chat_template.py" file for tests
    """
    repo_root = repo_root_dir()
    chat_template_path = (
        repo_root
        / "src"
        / "instructlab"
        / "training"
        / "chat_templates"
        / "ibm_generic_tmpl.py"
    )
    return chat_template_path


# TODO: This uses our data preprocessing utility which is not, itself, well tested.
# need to write tests for this as well.
@pytest.fixture(scope="module")
def cached_training_data(
    prepared_data_dir: pathlib.Path, cached_test_model: pathlib.Path
) -> pathlib.Path:
    """
    Renders test data in model template, tokenizes, and saves to filesystem.
    Subsamples NUM_SAMPLES_TO_KEEP examples to speed up tests.
    """

    data_in_repo = data_in_repo_path()
    chat_template = chat_template_in_repo_path()

    data_process_args = DataProcessArgs(
        data_output_path=str(prepared_data_dir),
        data_path=str(data_in_repo),
        max_seq_len=MINIMAL_TRAINING_ARGS["max_seq_len"],
        model_path=str(cached_test_model),
        chat_tmpl_path=str(chat_template),
        num_cpu_procs=RUNNER_CPUS_EXPECTED,
    )

    data_process.main(data_process_args)

    # Load the processed data and sample a subset
    output_path = prepared_data_dir / "data.jsonl"
    dataset = load_dataset("json", data_files=str(output_path), split="train")

    # Randomly sample NUM_SAMPLES_TO_KEEP examples
    sampled_dataset = dataset.shuffle(seed=42).select(
        range(min(NUM_SAMPLES_TO_KEEP, len(dataset)))
    )

    # Write the sampled data back to the same file
    sampled_dataset.to_json(str(output_path), num_proc=RUNNER_CPUS_EXPECTED)

    return output_path


@pytest.mark.slow
@pytest.mark.parametrize(
    "dist_backend", [DistributedBackend.FSDP, DistributedBackend.DEEPSPEED]
)
@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("lora_rank", [0])
@pytest.mark.parametrize("use_liger", [False, True])
def test_training_feature_matrix(
    cached_test_model: pathlib.Path,
    cached_training_data: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    prepared_data_dir: pathlib.Path,
    use_liger: bool,
    lora_rank: int,
    cpu_offload: bool,
    dist_backend: DistributedBackend,
) -> None:
    torch_args = TorchrunArgs(**DEFAULT_TORCHRUN_ARGS)
    train_args = TrainingArgs(
        model_path=str(cached_test_model),
        data_path=str(cached_training_data),
        data_output_dir=str(prepared_data_dir),
        ckpt_output_dir=str(checkpoint_dir),
        lora=LoraOptions(rank=lora_rank),
        use_liger=use_liger,
        **MINIMAL_TRAINING_ARGS,
    )

    train_args.distributed_backend = dist_backend

    if lora_rank > 0:
        # LoRA doesn't support full state saving.
        train_args.accelerate_full_state_at_epoch = False

    if dist_backend == DistributedBackend.FSDP:
        train_args.fsdp_options.cpu_offload_params = cpu_offload
    else:
        pytest.xfail("DeepSpeed not currently functional. OOMs during backprop.")
        if cpu_offload:
            pytest.xfail("DeepSpeed CPU Adam isn't currently building correctly")
        train_args.deepspeed_options.cpu_offload_optimizer = cpu_offload

    run_training(torch_args=torch_args, train_args=train_args)
