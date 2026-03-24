# On-Demand Checkpointing

On-demand checkpointing enables graceful checkpoint-and-exit when termination
signals are received during training. It is designed for environments like
OpenShift AI and KubeFlow where training jobs can be preempted at any time.

## How It Works

When enabled, the system installs signal handlers in the parent (launcher)
process that catch termination signals before the hard SIGKILL. When a signal
arrives:

1. The parent writes a trigger file to `/dev/shm` (a fast, node-local tmpfs).
2. Worker processes check for the trigger file at multiple synchronization
   points during each training step.
3. Workers coordinate via `all_reduce` so that if any rank on any node
   detects the trigger, all ranks agree to save.
4. A full-state checkpoint (model + optimizer + LR scheduler) is saved.
5. Workers exit cleanly.

On resume, the training loop detects the mid-epoch checkpoint, restores the
full training state, and fast-forwards to the exact step where training was
interrupted.

## Signals Handled

The following signals are intercepted (SIGKILL cannot be caught):

| Signal | Source |
|--------|--------|
| SIGTERM | Kubernetes graceful shutdown (default) |
| SIGINT | Ctrl-C / some job controllers |
| SIGUSR1 | Custom preemption controllers |
| SIGUSR2 | Custom preemption controllers |
| SIGXCPU | CPU time limit exceeded (resource quotas) |
| SIGHUP | Terminal disconnect / some eviction paths |

## Usage

### Python API

```python
from instructlab.training.config import TorchrunArgs, TrainingArgs
from instructlab.training import run_training

torch_args = TorchrunArgs(
    nproc_per_node=8,
    nnodes=1,
    node_rank=0,
    rdzv_id=12345,
    rdzv_endpoint="127.0.0.1:29500",
)

train_args = TrainingArgs(
    model_path="Qwen/Qwen2-1.5B-Instruct",
    data_path="./data.jsonl",
    data_output_dir="./processed",
    ckpt_output_dir="./checkpoints",
    num_epochs=3,
    on_demand_checkpointing=True,  # Enable the feature
    # ... other training args
)

run_training(torch_args, train_args)
```

### CLI

```bash
torchrun --nproc-per-node=8 \
    instructlab/training/main_ds.py \
    --model_name_or_path Qwen/Qwen2-1.5B-Instruct \
    --data_path ./data.jsonl \
    --output_dir ./checkpoints \
    --on_demand_checkpointing \
    ...
```

## Resume Behavior

When a checkpoint saved by on-demand checkpointing is found in the output
directory, the training loop automatically:

1. Loads the full optimizer and LR scheduler state from the checkpoint.
2. Reads `global_step` from the checkpoint metadata to determine where
   training was interrupted.
3. Resumes at the **same epoch** and fast-forwards to the exact step,
   skipping already-completed batches.

This differs from epoch-boundary checkpoints, which resume at the start of
the next epoch.

### Checkpoint Metadata

On-demand checkpoints store additional metadata compared to epoch-boundary
checkpoints:

```json
{
    "current_epoch": 0,
    "samples_seen": 144,
    "global_step": 19
}
```

The `global_step` field is what distinguishes an on-demand checkpoint from an
epoch-boundary one. When present, the resume logic keeps `current_epoch`
unchanged and sets `last_step = global_step` to enable fast-forwarding.

## Multi-Node Training

The trigger file mechanism works correctly across multiple nodes:

- The trigger file lives on `/dev/shm`, which is node-local. Each node's
  parent process writes its own trigger file when it receives a signal.
- Workers use `all_reduce(MAX)` to synchronize: if any rank on any node
  detects a trigger, all ranks on all nodes agree to save.
- The checkpoint itself is saved to the shared filesystem (the configured
  `ckpt_output_dir`), accessible by all nodes on resume.

## Stale Trigger Files

If a previous training run was killed before workers could clean up the
trigger file, the new run's `ParentSignalHandler` detects and removes it
during initialization. This prevents a new job from immediately
checkpointing and exiting due to a leftover trigger from a prior run.

## Manually Triggering a Checkpoint

You can trigger a checkpoint-and-exit without sending a signal by writing
the trigger file directly. This is useful for debugging, testing, or
integration with custom orchestration that doesn't use Unix signals.

The trigger file path is:

```
/dev/shm/instructlab_checkpoint_requested_<JOB_ID>
```

Where `<JOB_ID>` is the `rdzv_id` passed to `TorchrunArgs`. If no job ID
was set, the path is `/dev/shm/instructlab_checkpoint_requested` (no suffix).

To trigger a checkpoint from a shell on any node in the training cluster:

```bash
# Find the job ID (it's the rdzv_id, also stored in the environment)
JOB_ID=$(printenv INSTRUCTLAB_ON_DEMAND_JOB_ID)

# Write the trigger file
echo 1 > /dev/shm/instructlab_checkpoint_requested_${JOB_ID}
```

Or without the job ID:

```bash
echo 1 > /dev/shm/instructlab_checkpoint_requested
```

Workers check for the trigger file at each synchronization point in the
training loop (multiple times per step). Once any rank on any node detects
it, all ranks coordinate via `all_reduce` to save a checkpoint and exit.

You only need to write the file on **one node** — the `all_reduce` ensures
all nodes participate even if they don't see the file locally.

From Python:

```python
from instructlab.training.on_demand_checkpoint import write_trigger_file

write_trigger_file(job_id="12345")  # or job_id=None for default path
```

## Kubernetes / OpenShift Configuration

To give workers enough time to save a checkpoint before the hard SIGKILL,
increase `terminationGracePeriodSeconds` in your pod spec:

```yaml
spec:
  terminationGracePeriodSeconds: 300  # 5 minutes
```

The default of 30 seconds may not be enough for large models. The checkpoint
save time depends on model size, number of GPUs, and filesystem speed.
