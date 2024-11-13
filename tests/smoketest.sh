#!/usr/bin/env bash
set -eux -o pipefail

# ############### Read-only parameters ############### 
MODEL_NAME="instructlab/granite-7b-lab"
# gets directory of current file.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CORRECT_WORKING_DIR="${SCRIPT_DIR}/../src/instructlab/training/"
SAMPLE_DATA_PATH="${SCRIPT_DIR}/../sample-data/train_all_pruned_SDG.jsonl"
TMP_DIR=$(mktemp -d)
CHECKPOINTS_DIR="${TMP_DIR}/checkpoints"
DATA_DIR="${TMP_DIR}/data"
COMPUTED_DATA_PATH="${DATA_DIR}/data.jsonl"
DEFAULT_DISTRIB_FRAMEWORK='fsdp'
DISTRIB_FRAMEWORK="${1:-${DEFAULT_DISTRIB_FRAMEWORK}}" # defaults to FSDP
DEFAULT_GPUS=8
NUM_GPUS="${2:-${DEFAULT_GPUS}}"

# ############### User-modifiable parameters ############### 
# Change these as needed
MAX_BATCH_LEN=60000
MAX_SEQ_LEN=4096
NUM_SAMPLES_TRAINED_ON=5000 # upper-bound on training dataset size.

# ############### Test Functions ############### 

#######################################
# Creates directories for the precomputed datasets
# and the checkpoints that are saved during training inside
# of the temporary storage created for these tests.
# Globals:
#   CHECKPOINTS_DIR 
#   DATA_DIR
# Arguments:
#   None
# Returns:
#   None
#######################################
function setup_tmpdir () {
    mkdir "${CHECKPOINTS_DIR}"
    mkdir "${DATA_DIR}"
}

#######################################
# Test most common training parameters without using
# Flash Attention
# Globals:
#   SAMPLE_DATA_PATH
#   DATA_DIR
#   MODEL_NAME
#   NUM_SAMPLES_TRAINED_ON
#   COMPUTED_DATA_PATH
# Arguments:
#   None
# Returns:
#   echos number of samples trained on to standard out.
#######################################
function prepare_data () {
    # preprocesses .jsonl messages data so that it's a valid
    # input to the model (inputs tokenized, formatted with mask, etc.)
    # then, data is trimmed to a determined length to make training
    # go faster.
    
    python3 data_process.py \
    --data_path="${SAMPLE_DATA_PATH}" \
    --data_output_path="${DATA_DIR}" \
    --max_seq_len=4096 \
    --model_name_or_path="${MODEL_NAME}"

    # trim data so we only keep the first 'n' samples.
    # should be enough data for training to be meaningful but not enough
    # that training takes a large amount of time.
    echo "$(head -"${NUM_SAMPLES_TRAINED_ON}" "${COMPUTED_DATA_PATH}")" > "${COMPUTED_DATA_PATH}"

    echo "TRAINING ON $(wc -l "${COMPUTED_DATA_PATH}") SAMPLES"
}

#######################################
# Clears and remakes the temporary directory where
# artifacts, such as checkpoints and logs, are stored
# during training.
# Globals:
#   CHECKPOINTS_DIR
# Arguments:
#   None
# Returns:
#   writes location of checkpoints dir to standard out.
#######################################
function _cleanup_saved_checkpoints() {
    echo "CLEARING CHECKPOINTS: ${CHECKPOINTS_DIR}"
    rm -rf "${CHECKPOINTS_DIR}"
    mkdir "${CHECKPOINTS_DIR}"
}

#######################################
# Test most common training parameters without using
# Flash Attention
# Globals:
#   NUM_GPUS
#   MODEL_NAME
#   COMPUTED_DATA_PATH
#   CHECKPOINTS_DIR
#   DISTRIBUTED_FRAMEWORK
#   MAX_BATCH_LEN
# Arguments:
#   None
# Returns:
#   None
#######################################
function test_standard_loop () {
    torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    main_ds.py \
    --model_name_or_path="${MODEL_NAME}" \
    --data_path="${COMPUTED_DATA_PATH}" \
    --output_dir="${CHECKPOINTS_DIR}" \
    --num_epochs=1 \
    --effective_batch_size=128 \
    --save_samples=0 \
    --checkpoint_at_epoch \
    --accelerate_full_state_at_epoch \
    --distributed_training_framework="${DISTRIB_FRAMEWORK}" \
    --max_batch_len="${MAX_BATCH_LEN}" \
    --is_granite
}

#######################################
# Test most common training parameters without using
# Flash Attention
# Globals:
#   NUM_GPUS
#   MODEL_NAME
#   COMPUTED_DATA_PATH
#   CHECKPOINTS_DIR
#   DISTRIBUTED_FRAMEWORK
#   MAX_BATCH_LEN
# Arguments:
#   None
# Returns:
#   None
#######################################
function test_standard_loop_nongranite () {
    torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    main_ds.py \
    --model_name_or_path="${MODEL_NAME}" \
    --data_path="${COMPUTED_DATA_PATH}" \
    --output_dir="${CHECKPOINTS_DIR}" \
    --num_epochs=1 \
    --effective_batch_size=128 \
    --save_samples=0 \
    --checkpoint_at_epoch \
    --accelerate_full_state_at_epoch \
    --distributed_training_framework="${DISTRIB_FRAMEWORK}" \
    --max_batch_len="${MAX_BATCH_LEN}"
    # --is_granite \
}

#######################################
# Test most common training parameters without using
# Granite or Flash Attention
# Globals:
#   NUM_GPUS
#   MODEL_NAME
#   COMPUTED_DATA_PATH
#   CHECKPOINTS_DIR
#   DISTRIBUTED_FRAMEWORK
#   MAX_BATCH_LEN
# Arguments:
#   None
# Returns:
#   None
#######################################
function test_standard_loop_noflashattention_nogranite () {
    torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    main_ds.py \
    --model_name_or_path="${MODEL_NAME}" \
    --data_path="${COMPUTED_DATA_PATH}" \
    --output_dir="${CHECKPOINTS_DIR}" \
    --num_epochs=1 \
    --effective_batch_size=128 \
    --save_samples=0 \
    --checkpoint_at_epoch \
    --accelerate_full_state_at_epoch \
    --distributed_training_framework="${DISTRIB_FRAMEWORK}" \
    --max_batch_len="${MAX_BATCH_LEN}" \
    --disable_flash_attn
    # --is_granite
}


##############################################################################
# Validates the pathing logic for FSDP & LoRA.
# A valid run should result in a model with all adapters merged
# with the base model.
##############################################################################
function test_standard_loop_fsdp_lora() {
    torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    main_ds.py \
    --model_name_or_path="${MODEL_NAME}" \
    --data_path="${COMPUTED_DATA_PATH}" \
    --output_dir="${CHECKPOINTS_DIR}" \
    --num_epochs=1 \
    --effective_batch_size=128 \
    --save_samples=0 \
    --checkpoint_at_epoch \
    --distributed_training_framework="${DISTRIB_FRAMEWORK}" \
    --max_batch_len="${MAX_BATCH_LEN}" \
    --lora_r=4 \
    --lora_alpha=32 \
    --lora_dropout=0.1
}

function main () {

    setup_tmpdir
    trap 'rm -rf ${TMP_DIR}' EXIT

    #NOTE (jkunstle): script is run as though it's
    # in the same source dir as main_ds and data_process.
    cd "${CORRECT_WORKING_DIR}"
    echo "CURRENT WORKING DIRECTORY: $(pwd)"

    prepare_data
    test_standard_loop_noflashattention_nogranite
    _cleanup_saved_checkpoints
    test_standard_loop_nongranite
    _cleanup_saved_checkpoints
    test_standard_loop
    test_standard_loop_fsdp_lora
}

main
