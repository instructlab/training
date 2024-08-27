# About the InstructLab Training Library

<!-- update this whenever we update this document -->
*Date Written: August 27th, 2024*

We've written this document to help the reader understand the InstructLab training library
so they understand what the code is actually doing, and can make changes on their own as-needed.

This document was originally written on August 27th, 2024 and will most likely drift out of sync
over time. However; the core ideas & philosophy behind the decision choices made will likely
remain constant.

<!-- - very good to have a well documented codebase that we can talk about in the paper 
    - high level documentation that gives people a gist of what the code is doing
    - highlight that the code doesnt have any abstraction at all
    - want to see examples, something like a script or jupyter notebook -->


## Structure of the Training Library

The training library is very simple and exposes one primary function in its public API `instructlab.training:run_training`

The `run_training` function serves as the primary gateway through which users are able to
run a multitude of different training scenarios with ease. Its argument definition is as follows:

```py
def run_training(torch_args: TorchrunArgs, train_args: TrainingArgs) -> None:
    # implementation here
```

Through this simple API, one can go from training a quantized LoRA on a constrained system to a vast
multi-node distributed full fine-tune by changing only a few parameters.

The parmameters themselves are very straight-forward:
- `torch_args` contains the arguments which are passed when `torchrun` gets invoked on the training loop itself. This is where you would specify how many GPUs to utilize, what rendezvous endpoint each node should connect to, etc. 
- `train_args` controls the parameters of the training script itself. This is where you would specify values such as the learning rate, number of epochs to run, whether to use LoRA, quantization, CPU-offloading, etc.


We've documented these arguments elsewhere and so we won't cover them here so as to avoid misinforming the
reader with dated information. If you are interested, please see [README.md](./README.md) or the [config file](./src/instructlab/training/config.py) where they're implemented.



## How `run_training` works

The implementation of `run_training` itself is fairly simple: the [main_ds.py](./src/instructlab/training/main_ds.py) file is actually a script that one can invoke by directly calling `python` on it. It receives a number of flags and arguments which control its parameters. By default, the script is designed to run on only a single GPU.

When `run_training` is called, it uses the values supplied by `torch_args` to invoke the `main_ds.py` script
via the [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) CLI utility from PyTorch, whereas the `train_args` are passed as values to the training script itself. This causes the script to be invoked via PyTorch's elastic-run utility, where it invokes multiple copies of the `main_ds.py` file that each run on their own separate GPU. 

In a multi-node scenario, the rendezvous endpoint is also used to allow multiple nodes to network with the
primary runner node. The current implementation of training uses NCCL as the distributed backend.

### Overall structure of `run_training`

Here's a general view of what happens when `run_training` is invoked:

1. The training dataset provided through `train_args.data_path` is processed into a copy where all of the samples are tokenized and all samples where the number of tokens exceeds `train_args.max_seq_len` are dropped. The resultant dataset is written to an intermediate location, ideally into shared memory `/dev/shm` so that it's simpler for all of the subsequent processes to load.
1. The `train_args` are processed and formatted into their corresponding CLI args for `main_ds.py`
1. Main thread invokes the training script by calling `torchrun main_ds.py` with the `torch_args` and streams the output of the script
1. Within the training script, we perform setup such as initializing the distributed networking, loading datasets, tokenizers, etc.
1. With the dataset loaded, we analyze the state of our system (num GPUs, max batch length, etc.) and determine what sampler we can use (multipack vs. distributed)
1. We load the model with whatever parameters we decided, e.g. whether to quantize with BitsAndBytes, use flash-attention, load as a padding-free transformer, etc. 
    1. Depending on whether we use a padding-free transformer or not, we define the appropriate loss function
        * Padding-free requires us to adjust the tokens so that last token of one example doesn't predict the first token of the next example
        * The loss for all of the models needs to be reduced for all of the devices for proper backpropagation
    1. Noised embeddings are also added to the model
    1. LoRA is applied and wrapped as a peft model if specified.
    1. Enable gradient checkpointing
    1. [DeepSpeed] Select the optimizer to be either `FusedAdam` or `DeepSpeedCPUAdam` depending on whether CPU offloading is being used or not
    1. Create a learning-rate scheduler with our provided warmup steps, training states, and selected optimizer
    1. Wrap the model with DeepSpeed using the configured optimizer, scheduler, and DeepSpeed options
1. If we can resume training from a given checkpoint then do so
1. Adjust the rate of saving samples depending on our effective batch size (EBS)
1. Begin the training loop
    1. Iterate through each batch to compute the outputs and calculate loss
    1. Broadcast the loss across all of the nodes running the training
    1. Resize the loss by dividing it by the number of non-padding tokens and scaling it by the world size
    1. Each node then backpropagates the loss throughout the model
1. Set a distributed barrier so all nodes finish training before continuing




### 



