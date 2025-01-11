# Standard
import os
import pathlib
import shutil
import sys
import tempfile

# Third Party
from transformers import AutoModelForCausalLM
import huggingface_hub
import pytest

# First Party
from instructlab.training import data_process
from instructlab.training.config import (
    DataProcessArgs,
    DistributedBackend,
    TorchrunArgs,
    TrainingArgs,
)
from instructlab.training.main_ds import run_training

MINIMAL_TRAINING_ARGS = {
    "max_seq_len": 140,  # this config fits nicely on 4xL40s and may need modification for other setups
    "max_batch_len": 15000,
    "num_epochs": 1,
    "effective_batch_size": 3840,
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

REFERENCE_TEST_MODEL = "instructlab/granite-7b-lab"
RUNNER_CPUS_EXPECTED = 4

# matrix of training environments we'd like to test
DIST_BACKEND_FRAMEWORKS = ["fsdp", "deepspeed"]
USE_DOLOMITE = [True, False]
CPU_OFFLOADING = [True, False]
USE_LORA = [True, False]


@pytest.fixture(scope="module")
def custom_tmp_path():
    temp_dir = tempfile.mkdtemp()

    temp_path = pathlib.Path(temp_dir)

    yield temp_path

    shutil.rmtree(temp_path)


@pytest.fixture(scope="function")
def checkpoint_dir(custom_tmp_path: pathlib.Path) -> pathlib.Path:
    """
    Creates a 'checkpoints' directory for each test and deletes it afterward.
    """
    ckpt_dir = custom_tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    yield ckpt_dir

    shutil.rmtree(ckpt_dir)


@pytest.fixture(scope="module")
def prepared_data_dir(custom_tmp_path: pathlib.Path) -> pathlib.Path:
    data_file_dir = custom_tmp_path / "prepared_data"
    data_file_dir.mkdir()

    return data_file_dir


@pytest.fixture(scope="module")
def cached_model_dir(custom_tmp_path: pathlib.Path) -> pathlib.Path:
    model_dir = custom_tmp_path / "model"
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
        token=os.getenv("HF_TOKEN", None),
        repo_id=REFERENCE_TEST_MODEL,
        local_dir=cached_model_dir,
    )

    return cached_model_dir


def this_file_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve()


def data_in_repo_path() -> pathlib.Path:
    current_file_path = this_file_path()
    data_in_repo_path = (
        current_file_path.parents[2] / "sample-data" / "train_all_pruned_SDG.jsonl"
    )
    return data_in_repo_path


def chat_template_in_repo_path() -> pathlib.Path:
    current_file_path = this_file_path()
    chat_template_path = (
        current_file_path.parents[2]
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
    """Renders test data in model template, tokenizes, and saves to fs"""

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

    return prepared_data_dir / "data.jsonl"


@pytest.mark.skip
@pytest.mark.slow
def test_basic_training_run(
    cached_test_model: pathlib.Path,
    cached_training_data: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    prepared_data_dir: pathlib.Path,
) -> None:
    """
    Used for isolated test development. Skipped when not in use.
    """

    train_args = TrainingArgs(
        model_path=str(cached_test_model),
        data_path=str(cached_training_data),
        data_output_dir=str(prepared_data_dir),
        ckpt_output_dir=str(checkpoint_dir),
        **MINIMAL_TRAINING_ARGS,
    )

    torch_args = TorchrunArgs(**DEFAULT_TORCHRUN_ARGS)

    run_training(torch_args=torch_args, train_args=train_args)
    assert True


@pytest.mark.slow
@pytest.mark.parametrize("dist_backend", DIST_BACKEND_FRAMEWORKS)
@pytest.mark.parametrize("cpu_offload", CPU_OFFLOADING)
def test_training_feature_matrix(
    cached_test_model: pathlib.Path,
    cached_training_data: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    prepared_data_dir: pathlib.Path,
    cpu_offload: bool,
    dist_backend: str,
) -> None:
    train_args = TrainingArgs(
        model_path=str(cached_test_model),
        data_path=str(cached_training_data),
        data_output_dir=str(prepared_data_dir),
        ckpt_output_dir=str(checkpoint_dir),
        **MINIMAL_TRAINING_ARGS,
    )

    train_args.distributed_backend = DistributedBackend(dist_backend)
    if DistributedBackend.FSDP.value == dist_backend:
        train_args.fsdp_options.cpu_offload_params = cpu_offload
    else:
        pytest.xfail("DeepSpeed not currently functional. OOMs during backprop.")
        if cpu_offload:
            pytest.xfail("DeepSpeed CPU Adam isn't currently building correctly")
        train_args.deepspeed_options.cpu_offload_optimizer = cpu_offload

    torch_args = TorchrunArgs(**DEFAULT_TORCHRUN_ARGS)

    run_training(torch_args=torch_args, train_args=train_args)
