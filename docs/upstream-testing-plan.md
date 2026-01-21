# Testing in 'instructlab/instructlab'

We would like for this library to be a lightweight, efficient, and hackable toolkit for training LLMs with SFT and RLHF.

Currently, this library is used heavily by the `ilab` tool from `github.com/instructlab/instructlab`. It is tested via that project's end-to-end tests.
To improve development velocity, project trustworthiness, and library utility apart from `ilab`, we need to improve our project testing in-repo.

## Testing a training library

Training LLMs requires a lot of time and compute. We simply cannot "test" our library by executing an entire instruct-tuning run for each feature configuration we would like to verify.

This presents a challenge: we have to test our code as well as possible without frequently generating a complete final artifact (a "trained" model checkpoint).

To compensate for this challenge, we propose the following three-tier testing methodology:

0. Linting: dev-time verification that code meets style and static analysis requirements
1. Unit tests: verification that indivisible units work correctly
2. Subsystem tests: verification that collections of units work correctly together
3. Smoke tests: verification that feature-bounded systems work correctly
4. Benchmark tests: verification that outputs behave correctly

### Responsibilities of: Unit Tests

This is one of the most common patterns in software testing and requires little introduction.
For our purposes, Unit Tests will be tests that do not require accelerated hardware (e.g. GPUs).

These tests will verify that the smallest units of meaningful code work correctly.
They should be able to run on any development machine and should always run for incoming upstream PRs.

### Responsibilities of: Subsystem Tests

This might include data management utilities (distributed data sampler, batch collator, etc), model setup tools, loop instrumentation tools, etc.
Subsystem Tests should give very fast confirmation that the library is behaving correctly, especially when bumping dependency versions.

Subsystem Tests should be able to run on any development machine and should always run for incoming upstream PRs.

### Responsibilities of: Smoke Tests

The origin of the term "smoke test" comes from plumbing. Plumbers would pipe smoke, instead of water, through a completed pipe system to check that there were no leaks.
In computer software, smoke testing provides fast verification to decide whether further testing is necessary- if something immediately breaks, we quickly know that there's a
problem, and shouldn't spend resources on more in-depth testing.

This is effectively what we'd like to do with this class of tests: verify that everything _seems_ to be in order before actually building a model to verify this.

The challenge, however, is that "basic" testing in this context probably still requires accelerated hardware. We can't check that checkpointing a LoRA model while using CPU Offloading
doesn't break without spinning up that exact testing scenario.

"Smoke tests" in this repo, therefore, will be defined as tests that demonstrate the functionality, but don't prove the final model correctness, for features that we want to ship. This is
very helpful to us because _most important failures don't happen during training itself- they happen during setup, checkpointing, and tear-down_.

Pictorially, we can represent smoke testing as abbreviating a training run ('0' are setup/tear-down/checkpointing steps, '.' are training steps)"

00000.................................................00000.....................................................00000

to the following:

00000.00000.00000

Smoke tests would block PRs from being merged and would run automatically once all unit and lint tests pass.
This puts the onus on the Unit Tests (and the test writer) to verify code correctness as much as they possibly can without using hardware. If something requires many invocations of
smoke tests to debug, it probably wasn't sufficiently debugged during development, or is insufficiently unit tested.

Smoke tests will inevitably require a high-spec development machine to run locally. It shouldn't be acceptable for smoke tests to run for >60 minutes- we should aim for them to run in <30 minutes.

#### Smoke tests coverage

This library attempts to provide training support for multiple variants of accelerators, numbers of accelerators, two distributed backends, and many features. Therefore, the completeness matrix
is roughly explained by the following pseudocode:

```python

for variant in ["nvidia", "amd", "intel"]: # parallel at runner level
    for framework in ["fsdp", "deepspeed"]:
        for n_card in [1, 4]:
            for feat_perm in all_permutations(["lora", "cpu_offload", None, ...]):

                opts = [variant, n_card, framework, *feat_perm]

                if check_opts_compatible(*opts):
                    run_accelerated_tests(*opts)

```

When we have to add a new accelerator variant or feature, we can plug them into their respective tier in the smoke testing hierarchy.

### Responsibilities of: Benchmark Tests

Intermittently, we'll want to verify that our "presumeably correct" training library _actually produces_ a high-performance model.
Predicting model performance from statically analyzing the logits or the parameters isn't viable, so it'll have to be benchmarked.

Benchmark testing might look like the following:

1. Tune a few models on well-understood dataset using one or two permutations of features, unique to benchmark run.
2. Benchmark model on battery of evaluation metrics that give us the strongest signal re: model performance.

Implementing this testing schema can be another project for a later time.

## Concerns and Notes

### Running smoke tests is expensive and shouldn't be done for every contribution

Repository maintainers may want to implement a manual smoke-testing policy based on what a PR actually changes, rather than have the test be automatic. If an incoming PR changes the library interface but doesn't directly affect the training code, a unit test would probably be the most appropriate way to check whether all behavior still squares.

### How often should we be running benchmark tests?

There are many options for this depending on the observed risk that new features or dependency bumps seem to bring to the project. Likely, a comprehensive benchmark test should be done as a baseline and then the test should only be repeated when new model architectures are supported, major versions of ilab SDG are released, etc.

### How should these tests be implemented? Bash scripts? Pytest?

Pytest. Pytest + Tox. Writing tests in Bash is very hard to maintain.

## Conclusion

Testing ML models has never been easy, and it's unlikely that the final implementation of these ideas will closely mirror this document. However, we hope that the _spirit_ of this document- fast feedback, efficient use of resources, unit testing prioritization- will be effective.
