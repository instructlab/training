# InstructLab Training Library

![Lint](https://github.com/instructlab/training/actions/workflows/lint.yml/badge.svg?branch=main)
![Build](https://github.com/instructlab/training/actions/workflows/pypi.yaml/badge.svg?branch=main)
![Release](https://img.shields.io/github/v/release/instructlab/training)
![License](https://img.shields.io/github/license/instructlab/training)

![`e2e-nvidia-l4-x1.yml` on `main`](https://github.com/instructlab/training/actions/workflows/e2e-nvidia-l4-x1.yml/badge.svg?branch=main)
![`e2e-nvidia-l40s-x4.yml` on `main`](https://github.com/instructlab/training/actions/workflows/e2e-nvidia-l40s-x4.yml/badge.svg?branch=main)

## About the Library

The InstructLab Training library is an optimized model instruction-tuning library, designed for messages-format data. This library can be used for efficiently fine-tuning Causal Language Models, working for both base models and previously-aligned models with existing chat templates. This library was used to achieve the results found in [Unveiling the Secret Recipe: A Guide For Supervised Fine-Tuning Small LLMs](https://arxiv.org/abs/2412.13337).

To simplify the process of fine-tuning models with the [LAB
method](https://arxiv.org/abs/2403.01081), or for general use, this library provides a simple pythonic training interface.

## Usage and Guidance Sections

- [Installing](#installing-the-library)
  - [Additional Nvidia packages](#additional-nvidia-packages)
- [Using the library](#using-the-library)
- [Learning about the training arguments](#learning-about-training-arguments)
  - [`TrainingArgs`](#trainingargs)
  - [`DeepSpeedOptions`](#deepspeedoptions)
  - [`FSDPOptions`](#fsdpoptions)
  - [`loraOptions`](#loraoptions)
- [Learning about `TorchrunArgs` arguments](#learning-about-torchrunargs-arguments)
- [Example training run with arguments](#example-training-run-with-arguments)

## Installing the library

To get started with the library, you must clone this repository and install it via `pip`.

Install the library:

```bash
pip install instructlab-training 
```

You can then install the library for development:

```bash
pip install -e ./training
```

### Additional NVIDIA packages

This library uses the `flash-attn` package as well as other packages, which rely on NVIDIA-specific CUDA tooling to be installed.
If you are using NVIDIA hardware with CUDA, you need to install the following additional dependencies.

Basic install

```bash
pip install .[cuda]
```

Editable install (development)

```bash
pip install -e .[cuda]
```

## Using the library

See the `examples` dir for guided sample notebooks on library usage. Below provides some added details on library options:

You can utilize this training library by importing the necessary items.

```py
from instructlab.training import (
    run_training,
    TorchrunArgs,
    TrainingArgs,
    DeepSpeedOptions
)
```

You can then define various training arguments. They will serve as the parameters for your training runs. See:

- [Learning about the training argument](#learning-about-training-arguments)
- [Example training run with arguments](#example-training-run-with-arguments)

## Learning about training arguments

The `TrainingArgs` class provides most of the customization options
for training jobs. There are a number of options you can specify, such as setting
`DeepSpeed` config values or running a `LoRA` training job instead of a full fine-tune.

### `TrainingArgs`

| Field | Description |
| --- | --- |
| model_path | Either a reference to a HuggingFace repo or a path to a model saved in the HuggingFace format.  |
| data_path | A path to the `.jsonl` training dataset. This is expected to be in the messages format.  |
| ckpt_output_dir | Directory where trained model checkpoints will be saved. |
| data_output_dir | Directory where the processed training data is stored (post filtering/tokenization/masking) |
|  max_seq_len | The maximum sequence length to be included in the training set. Samples exceeding this length will be dropped. |
| max_batch_len | Maximum tokens per gpu for each batch that will be handled in a single step. Used as part of the multipack calculation. If running into out-of-memory errors, try to lower this value, but not below the `max_seq_len`. |
| num_epochs | Number of epochs to run through before stopping. |
| effective_batch_size | The amount of samples in a batch to see before we update the model parameters. |
| save_samples | Number of samples the model should see before saving a checkpoint. Consider this to be the checkpoint save frequency. |
| learning_rate | How fast we optimize the weights during gradient descent. Higher values may lead to unstable learning performance. It's generally recommended to have a low learning rate with a high effective batch size. |
| warmup_steps | The number of steps a model should go through before reaching the full learning rate. We start at 0 and linearly climb up to `learning_rate`. |
| random_seed | The random seed PyTorch will use. |
| mock_data | Whether or not to use mock, randomly generated,  data during training. For debug purposes |
| mock_data_len | Max length of a single mock data sample. Equivalent to `max_seq_len` but for mock data. |
| deepspeed_options | Config options to specify for the DeepSpeed optimizer. |
| lora | Options to specify if you intend to perform a LoRA train instead of a full fine-tune. |
| chat_tmpl_path | Specifies the chat template / special tokens for training. |
| checkpoint_at_epoch | Whether or not we should save a checkpoint at the end of each epoch. |
| fsdp_options | The settings for controlling FSDP when it's selected as the distributed backend. |
| distributed_backend | Specifies which distributed training backend to use. Supported options are "fsdp" and "deepspeed". |
| disable_flash_attn | Disables flash attention when set to true. This allows for training on older devices. |
| keep_last_checkpoint_only | Determines whether we should only keep the last checkpoint directory - the previous checkpoint directory is always overwritten. The checkpoint directory is called `last_epoch`. |

### `DeepSpeedOptions`

This library only currently support a few options in `DeepSpeedOptions`:
The default is to run with DeepSpeed, so these options only currently
allow you to customize aspects of the ZeRO stage 2 optimizer.

| Field | Description |
| --- | --- |
| cpu_offload_optimizer | Whether or not to do CPU offloading in DeepSpeed stage 2. |
| cpu_offload_optimizer_ratio | Floating point between 0 & 1. Specifies the ratio of parameters updating (i.e. optimizer step) on CPU side. |
| cpu_offload_optimizer_pin_memory | If true, offload to page-locked CPU memory. This could boost throughput at the cost of extra memory overhead. |
| save_samples | The number of samples to see before saving a DeepSpeed checkpoint. |

For more information about DeepSpeed, see [deepspeed.ai](https://www.deepspeed.ai/)

#### DeepSpeed with CPU Offloading

To use DeepSpeed with CPU offloading, you'll usually encounter an issue indicating that the optimizer needed to use the Adam optimizer on CPU doesn't exist. To resolve this, please follow the following steps:

**Rebuild DeepSpeed with CPUAdam**:

You'll need to rebuild DeepSpeed in order for the optimizer to be present:

```bash
# uninstall deepspeed & reinstall with the flags for installing CPUAdam
pip uninstall deepspeed
DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install deepspeed --no-deps
```

**Ensure `-lcurand` is linked correctly**:

A problem that we commonly encounter is that the `-lcurand` linker will not be present when
DeepSpeed recompiles. To resolve this, you will need to find the location of the `libcurand.so` file in your machine:

```bash
find / -name 'libcurand*.so*' 2>/dev/null
```

If libcurand.so is not present in the `/usr/lib64` directory, you'll need to add a symlink. Ex:

```bash
sudo ln -s /usr/local/cuda/lib64/libcurand.so.10 /usr/lib64/libcurand.so
```

### `FSDPOptions`

Like DeepSpeed, we only expose a number of parameters for you to modify with FSDP.
They are listed below:

| Field | Description |
| --- | --- |
| cpu_offload_params | When set to true, offload parameters from the accelerator onto the CPU. This is an all-or-nothing option. |
| sharding_strategy | Specifies the model sharding strategy that FSDP should use. Valid options are:  `FULL_SHARD` (ZeRO-3), `HYBRID_SHARD` (ZeRO-3*), `SHARD_GRAD_OP` (ZeRO-2), and `NO_SHARD`. |

> [!NOTE]
> For `sharding_strategy` - Only `SHARD_GRAD_OP` has been extensively tested and is actively supported by this library.

### `loraOptions`

LoRA options currently supported:

| Field | Description |
| --- | --- |
| rank | The rank parameter for LoRA training. |
| alpha | The alpha parameter for LoRA training. |
| dropout | The dropout rate for LoRA training. |
| target_modules | The list of target modules for LoRA training. |
| quantize_data_type | The data type for quantization in LoRA training. Valid options are `None` and `"nf4"` |

#### Example run with LoRa options

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

### Learning about `TorchrunArgs` arguments

When running the training script, we always invoke `torchrun`.

If you are running a single-GPU system or something that doesn't
otherwise require distributed training configuration, you can create a default object:

```python
run_training(
    torchrun_args=TorchrunArgs(),
    training_args=TrainingArgs(
        # ...
    ),
)
```

However, if you want to specify a more complex configuration,
the library currently supports all the options that [torchrun accepts
today](https://pytorch.org/docs/stable/elastic/run.html#definitions).

> [!NOTE]
> For more information about the `torchrun` arguments, please consult the [torchrun documentation](https://pytorch.org/docs/stable/elastic/run.html#definitions).

#### Example training run with `TorchrunArgs` arguments

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

## Example training run with arguments

Define the training arguments which will serve as the
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

## Example training with separate data pre-processing

If the machines in the example above have shared storage, users can pre-process the training dataset a single time so that it can then be distributed to each machine by making the following updates.

```python
from instructlab.training import (
    run_training,
    TorchrunArgs,
    TrainingArgs,
    DeepSpeedOptions,
    DataProcessArgs,
    data_process as dp
)

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
    random_seed = 42,
    process_data = True,
)
...

data_process_args = DataProcessArgs(
    data_output_path = training_args.data_output_dir,
    model_path = training_args.model_path,
    data_path = training_args.data_path,
    max_seq_len = training_args.max_seq_len,
    chat_tmpl_path =  training_args.chat_tmpl_path
)

dp.main(data_process_args)
run_training(
    torch_args=torchrun_args,
    train_args=training_args,
)
```
