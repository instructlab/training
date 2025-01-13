import pytest
import pathlib
import tempfile
import shutil
import os
from instructlab.training.config import TrainingArgs, TorchrunArgs, DataProcessArgs
from instructlab.training.main_ds import run_training
from instructlab.training import data_process
from transformers import AutoModelForCausalLM

MINIMAL_TRAINING_ARGS = {
    "max_seq_len": 4096,
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
    "data_output_dir": "",  # will preprocess and cache the data ahead of time.
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
DIST_TRAIN_FRAMEWORKS = ["fsdp", "deepspeed"]
USE_DOLOMITE = [True, False]
CPU_OFFLOADING = [True, False]
USE_LORA = [True, False]


@pytest.fixture(scope="module")
def my_tmp_path():
    my_temp_dir = tempfile.mkdtemp()

    my_temp_path = pathlib.Path(my_temp_dir)

    yield my_temp_path

    print("CLEANING UP TEMP DIR")

    shutil.rmtree(my_temp_path)


@pytest.fixture(scope="module")
def checkpoint_dir(my_tmp_path: pathlib.Path) -> pathlib.Path:
    return my_tmp_path / "checkpoints"


@pytest.fixture(scope="module")
def prepared_data_file(my_tmp_path: pathlib.Path) -> pathlib.Path:
    data_file_loc = my_tmp_path / "prepped_data" / "data.jsonl"
    return data_file_loc


@pytest.fixture(scope="module")
def cached_model_dir(my_tmp_path: pathlib.Path) -> pathlib.Path:
    return my_tmp_path / "model"


# TODO: It's wasteful to redownload the model parameters for each test run.
# Could bake it into the test instance AMI instead and always
# try to load the model from storage before attempting to download it.
@pytest.fixture(scope="module")
def cached_7b_model(cached_model_dir: pathlib.Path) -> pathlib.Path:
    """Downloads our reference test model to temporary cache."""

    os.environ["HF_HOME"] = str(cached_model_dir)
    model = AutoModelForCausalLM.from_pretrained(REFERENCE_TEST_MODEL)

    del model  # don't need the actual model that was instantiated.

    return cached_model_dir


# TODO: This uses our data preprocessing utility which is not, itself, well tested.
# need to write tests for this as well.
@pytest.fixture(scope="module")
def cached_training_data(
    prepared_data_file: pathlib.Path, cached_7b_model: pathlib.Path
) -> pathlib.Path:
    """Renders test data in model template, tokenizes, and saves to fs"""

    current_file_path = pathlib.Path(__file__).resolve()
    data_in_repo = (
        current_file_path.parents[1] / "sample-data" / "train_all_pruned_SDG.jsonl"
    )
    chat_template = (
        current_file_path.parents[1]
        / "src"
        / "instructlab"
        / "training"
        / "chat_templates"
        / "ibm_generic_tmpl.py"
    )

    data_process_args = DataProcessArgs(
        data_output_path=str(prepared_data_file),
        data_path=str(data_in_repo),
        max_seq_len=MINIMAL_TRAINING_ARGS["max_seq_len"],
        model_path=REFERENCE_TEST_MODEL,
        chat_tmpl_path=str(chat_template),
        num_cpu_procs=RUNNER_CPUS_EXPECTED,
    )

    data_process.main(data_process_args)

    return prepared_data_file


@pytest.mark.slow
def test_basic_training_run(
    cached_7b_model: pathlib.Path,
    cached_training_data: pathlib.Path,
    checkpoint_dir: pathlib.Path,
    prepared_data_file: pathlib.Path,
) -> None:
    """
    Test that runs training with as many features
    turned off as possible. Meant to test the shortest, simplest training path.
    """

    train_args = TrainingArgs(
        model_path=str(cached_7b_model),
        data_path=str(cached_training_data),
        ckpt_output_dir=str(prepared_data_file),
        **MINIMAL_TRAINING_ARGS,
    )

    torch_args = TorchrunArgs(**DEFAULT_TORCHRUN_ARGS)

    run_training(torch_args=torch_args, train_args=train_args)


@pytest.mark.skip
@pytest.mark.slow
@pytest.mark.parametrize("cpu_offload", CPU_OFFLOADING)
@pytest.mark.parametrize("use_lora", USE_LORA)
@pytest.mark.parametrize("use_dolomite", USE_DOLOMITE)
@pytest.mark.parametrize("dist_train_framework", DIST_TRAIN_FRAMEWORKS)
def test_training_runs(
    dist_train_framework: str, use_dolomite: bool, use_lora: bool, cpu_offload: bool
) -> None: ...
