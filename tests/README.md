## Overview

`smoketest.sh` cd's into the source directory and runs the script-entrypoint for `main_ds.py` and `data_process.py`. Testing will break if file names or locations in the source tree change.

Existing tests are "smoke tests," meant to demonstrate that training completes (returns 0) or not. This is helpful to check if all required dependencies are installed.

Current tests add features as they go:

1. No Flash Attention or Granite
2. No Granite but Flash Attention enabled
3. Granite and Flash Attention enabled

## Usage

The testing script can be run without parameters as `./smoketest.sh`. By default, this will run all tests with `FSDP` as the distributed training backend. To change the distributed training backend to the other available option, one can run the script as `./smoketest.sh deepspeed`.

The second positional argument is for "number of GPUs"- e.g.: `./smoketest.sh fsdp 8`. This will run the test with 8 GPUs with fsdp as the distributed backend.

> [!NOTE]
> You'll need to install the training library to run the test. Inside a virtual environment and at inside the repo, please run `pip3 install -e .` to install the package in editable mode.
