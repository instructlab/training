# InstructLab Training Library

![Lint](https://github.com/instructlab/training/actions/workflows/lint.yml/badge.svg?branch=main)
![Build](https://github.com/instructlab/training/actions/workflows/pypi.yaml/badge.svg?branch=main)
![Release](https://img.shields.io/github/v/release/instructlab/training)
![License](https://img.shields.io/github/license/instructlab/training)

In order to simplify the process of fine-tuning models through the LAB
method, this library provides a simple training interface.

## Installation

To get started with the library, you must clone this repo and install it from source via `pip`:

```bash
# clone the repo and switch to the directory
git clone https://github.com/instructlab/training
cd training

# install the library
pip install .
```

For development, install it instead with `pip install -e .` instead
to make local changes while using this library elsewhere.

### Installing Additional NVIDIA packages

We make use of `flash-attn` and other packages which rely on NVIDIA-specific
CUDA tooling to be installed.

If you are using NVIDIA hardware with CUDA, please install the additional dependencies via:

```bash
# for a regular install
pip install .[cuda]

# or, for an editable install (development)
pip install -e .[cuda]
```

## Usage

Using the library is fairly straightforward, import the necessary items,

```py
from instructlab.training import (
    run_training,
    TorchrunArgs,
    TrainingArgs,
    DeepSpeedOptions
)
```

Then, define the training arguments which will serve as the
parameters for our training run:

```py
# define training-specific arguments
training_args = TrainingArgs(
    # define data-specific arguments
    model_path = "ibm-granite/granite-7b-base",
    data_path = "path/to/dataset.jsonl",
    ckpt_output_dir = "data/saved_checkpoints",
    data_output_dir = "data/outputs",

    # define model-trianing parameters
    max_seq_len = 4096,
    max_batch_len = 60000,
    num_epochs = 10,
    effective_batch_size = 3840,
    save_samples = 250000,
    learning_rate = 2e-6,
    warmup_steps = 800,
    is_padding_free = True, # set this to true when using Granite-based models
    random_seed = 42,
)
```

We'll also need to define the settings for running a multi-process job
via `torchrun`. To do this, create a `TorchrunArgs` object.

> [!TIP]
> Note, for single-GPU jobs, you can simply set `nnodes = 1` and `nproc_per_node=1`.

```py
torchrun_args = TorchrunArgs(
    nnodes = 1, # number of machines 
    nproc_per_node = 8, # num GPUs per machine
    node_rank = 0, # node rank for this machine
    rdzv_id = 123,
    rdzv_endpoint = '127.0.0.1:12345'
)
```

Finally, you can just call `run_training` and this library will handle
the rest 🙂.

```py
run_training(
    torchrun_args=torchrun_args,
    training_args=training_args,
)

```

### Customizing `TrainingArgs`

The `TrainingArgs` class provides most of the customization options
for the training job itself. There are a number of options you can specify, such as setting
DeepSpeed config values or running a LoRA training job instead of a full fine-tune.

Here is a breakdown of the general options:

| Field | Description |
| --- | --- |
| model_path | Either a reference to a HuggingFace repo or a path to a model saved in the HuggingFace format.  |
| data_path | A path to the `.jsonl` training dataset. This is expected to be in the messages format.  |
| ckpt_output_dir | Directory where trained model checkpoints will be saved. |
| data_output_dir | Directory where we'll store all other intermediary data such as log files, the processed dataset, etc. |
|  max_seq_len | The maximum sequence length to be included in the training set. Samples exceeding this length will be dropped. |
| max_batch_len | The maximum length of all training batches that we intend to handle in a single step. Used as part of the multipack calculation. If running into out-of-memory errors, try to lower this value, but not below the `max_seq_len`. |
| num_epochs | Number of epochs to run through before stopping. |
| effective_batch_size | The amount of samples in a batch to see before we update the model parameters. Higher values lead to better learning performance. |
| save_samples | Number of samples the model should see before saving a checkpoint. Consider this to be the checkpoint save frequency. The amount of storage used for a single training run will usually be `4GB * len(dataset) / save_samples` |
| learning_rate | How fast we optimize the weights during gradient descent. Higher values may lead to unstable learning performance. It's generally recommended to have a low learning rate with a high effective batch size. |
| warmup_steps | The number of steps a model should go through before reaching the full learning rate. We start at 0 and linearly climb up to `learning_rate`. |
| is_padding_free | Boolean value to indicate whether or not we're training a padding-free transformer model such as Granite. |
| random_seed | The random seed PyTorch will use. |
| mock_data | Whether or not to use mock, randomly generated,  data during training. For debug purposes |
| mock_data_len | Max length of a single mock data sample. Equivalent to `max_seq_len` but for mock data. |
| deepspeed_options | Config options to specify for the DeepSpeed optimizer. |
| lora | Options to specify if you intend to perform a LoRA train instead of a full fine-tune. |

#### `DeepSpeedOptions`

We only currently support a few options in `DeepSpeedOptions`:
The default is to run with DeepSpeed, so these options only currently
allow you to customize aspects of the ZeRO stage 2 optimizer.

| Field | Description |
| --- | --- |
| cpu_offload_optimizer | Whether or not to do CPU offloading in DeepSpeed stage 2. |

#### `loraOptions`

If you'd like to do a LoRA train, you can specify a LoRA
option to `TrainingArgs` via the `LoraOptions` object.

```python
from instructlab.training import LoraOptions, TrainingArgs

training_args = TrainingArgs(
    lora = LoraOptions(
        rank = 4,
        alpha = 32,
        dropout = 0.1,
    ),
    # ...
)
```

Here is the definition for what we currently support today:

| Field | Description |
| --- | --- |
| rank | The rank parameter for LoRA training. |
| alpha | The alpha parameter for LoRA training. |
| dropout | The dropout rate for LoRA training. |
| target_modules | The list of target modules for LoRA training. |
| quantize_data_type | The data type for quantization in LoRA training. Valid options are `None` and `"nf4"` |

### Customizing `TorchrunArgs`

When running the training script, we always invoke `torchrun`.

If you are running a single-GPU system or something that doesn't
otherwise require distributed training configuration, you can
just create a default object:

```python
run_training(
    torchrun_args=TorchrunArgs(),
    training_args=TrainingArgs(
        # ...
    ),
)
```

However, if you want to specify a more complex configuration,
we currently expose all of the options that [torchrun accepts
today](https://pytorch.org/docs/stable/elastic/run.html#definitions).

> ![NOTE]
> For more information about the `torchrun` arguments, please consult the [torchrun documentation](https://pytorch.org/docs/stable/elastic/run.html#definitions).

For example, in a 8-GPU, 2-machine system, we would
specify the following torchrun config:

```python
MASTER_ADDR = os.getenv('MASTER_ADDR')
MASTER_PORT = os.getnev('MASTER_PORT')
RDZV_ENDPOINT = f'{MASTER_ADDR}:{MASTER_PORT}'

# on machine 1
torchrun_args = TorchrunArgs(
    nnodes = 2, # number of machines 
    nproc_per_node = 4, # num GPUs per machine
    node_rank = 0, # node rank for this machine
    rdzv_id = 123,
    rdzv_endpoint = RDZV_ENDPOINT
)

run_training(
    torchrun_args=torchrun_args,
    training_args=training_args
)
```

```python
MASTER_ADDR = os.getenv('MASTER_ADDR')
MASTER_PORT = os.getnev('MASTER_PORT')
RDZV_ENDPOINT = f'{MASTER_ADDR}:{MASTER_PORT}'

# on machine 2
torchrun_args = TorchrunArgs(
    nnodes = 2, # number of machines 
    nproc_per_node = 4, # num GPUs per machine
    node_rank = 1, # node rank for this machine
    rdzv_id = 123,
    rdzv_endpoint = f'{MASTER_ADDR}:{MASTER_PORT}'
)

run_training(
    torch_args=torchrun_args,
    train_args=training_args
)
```
