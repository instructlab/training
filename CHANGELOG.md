# Changelog

## v0.5.5

### Features
* e2e: replace old small job with new medium job

### Fixes
* fix: incorrect label for AWS medium runner
* chore: add exit code & tox fix

### Infrastructure
* ci: grant HF_TOKEN access to the medium-size E2E CI job

## v0.5.4

### Features
* Add rocm extra to pyproject.toml

## v0.5.3

### Fixes
* fix: Add explicit flash_attn requirement for ROCm

## v0.5.2 - Fix Pretraining Masking

### Fixes
* fix: improve linting and automation
* Fix pretrain token list->int for masking

## v0.5.1

### Fixes
* fix: updates sorting logic to correctly compare numbers

## v0.5.0 - FSDP and Full-State Checkpoint Resuming

### Features
* feat: add e2e test for instructlab CI
* feat: add mergify
* Adding FSDP Support to Training Library by @aldopareja @Maxusmusti @RobotSail
* adds Accelerate full-state (opt, lr_sched, params)
* changes StreamablePopen to return a process and implement listening

### Fixes
* Fix lint error to make CI happy
* Fix typos
* Ap/fix multipack for non granite models
* Fix generic chat template saved to tokenizer for generation
* Fix linting error and missing quote

### Infrastructure
* Add license identifiers
* ci: update runner labels to uniquely identify instance sizes
* ci: minor cleanup of E2E job
* Fixing e2e to use relative path for working-directory
* switch -T to -a
* github: add stale bot to training repo
* fix: markdown lint error and mergify bug
* Bump actions/checkout from 4.1.7 to 4.2.0
* Bump step-security/harden-runner from 2.8.1 to 2.9.1
* Bump pypa/gh-action-pypi-publish from 1.9.0 to 1.10.2
* Bump actions/setup-python from 5.1.0 to 5.2.0
* Bump rhysd/actionlint from 1.7.1 to 1.7.2
* Bump hynek/build-and-inspect-python-package from 2.6.0 to 2.9.0
* Bump DavidAnson/markdownlint-cli2-action from 16.0.0 to 17.0.0
* ci: fix lint action
* ci: add AWS tags to show github ref and PR num for all jobs

## v0.5.0 Alpha 0 - The FSDP Release Pre-release

### Description
The FSDP Release introduces FSDP support in addition to the existing DeepSpeed support through the accelerate library.

### Features
* feat: add e2e test for instructlab CI
* feat: add mergify
* Adding FSDP Support to Training Library by @aldopareja @Maxusmusti @RobotSail

### Fixes
* Fix lint error to make CI happy
* Fix typos
* Ap/fix multipack for non granite models
* Fix linting error and missing quote

### Infrastructure
* Add license identifiers
* ci: update runner labels to uniquely identify instance sizes
* ci: minor cleanup of E2E job
* Fixing e2e to use relative path for working-directory
* Bump step-security/harden-runner from 2.8.1 to 2.9.1
* Bump pypa/gh-action-pypi-publish from 1.9.0 to 1.10.2
* Bump actions/setup-python from 5.1.0 to 5.2.0
* Bump rhysd/actionlint from 1.7.1 to 1.7.2
* Bump hynek/build-and-inspect-python-package from 2.6.0 to 2.9.0
* Bump DavidAnson/markdownlint-cli2-action from 16.0.0 to 17.0.0
* ci: fix lint action
* ci: add AWS tags to show github ref and PR num for all jobs

## v0.4.2

### Features
* Provide safeguards during training

## v0.4.1

### Changes
* makes saving every save_samples an optional feature

## v0.4.0

### Features
* Adds a flag to save checkpoints at the end of an epoch

### Changes
* Change success message at end of training

## v0.3.2

### Features
* Accept tuples for lora.target_modules

### Documentation
* patch some hyper parameter arg descriptions in README

## v0.3.1

### Dependencies
* Update requirements to have bitsandbytes min and dolomite min

## v0.3.0

### Features
* Updating token masking to support pretraining w/ masked special tokens
* Adding weight merging for LoRA/QLoRA ckpts

### Fixes
* remove dead code
* fix: changes the check to check against both the enum option and enum value

## v0.2.0

### Features
* Fix ckpt save to include architecture for inference runtime consumption
* Logging updates

### Performance
* Reducing deepspeed timeout to 10mins

## v0.1.0

### Features
* Flash Attention Disable Toggle (Take 2)

### Performance
* Reduce Unnecessary Multiprocessing

### Fixes
* 🐛: fix optimizer selection logic so that FusedAdam is never loaded when CPU offloading is enabled
* Add wheel to requirements

## v0.0.5.1

### Fixes
This release includes PR [#121](https://github.com/instructlab/training/pull/121) to overcome an issue where our way of lazily importing the run_training function is being picked up as an error by pylint.

## v0.0.5
Minor bugfixes and updates.

## v0.0.4
Minor bugfixes and updates.

## v0.0.3
Minor bugfixes and updates.

## v0.0.2

### Features
This introduces the instructlab library as a package in the instructlab package namespace.

To install it:
```
pip install instructlab-training
```

And to install it with flash-attn and other CUDA-dependent packages, you can use
```
pip install instructlab-training[cuda]
```

Here's how to use it:
```python
from instructlab.training.config import TorchrunArgs, TrainingArgs, run_training

torchrun_args = TorchrunArgs(
    nproc_per_node = 1,  # 1 GPU
    nnodes = 1,  # only 1 overall machine in the system
    node_rank = 0,  # rank of the current machine
    rdzv_id = 123,  # what ID other nodes will join on
    rdzv_endpoint = '0.0.0.0:12345'  # address where other nodes will join
)

training_args = TrainingArgs(
    # specify training args here
)

run_training(torch_args = torchrun_args, train_args = training_args)
```

## v0.0.1

### Features
Initial release with same features as v0.0.2.