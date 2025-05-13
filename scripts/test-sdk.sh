#!/usr/bin/env bash
set -xeuf

# generic globals
BOLD='\033[1m'
NC='\033[0m' # No Color
PRESERVE=0

# path and token globals
SCRIPTDIR=$(dirname "$0")
E2E_TEST_DIR=""
CONFIG_HOME=""
DATA_HOME=""
CACHE_HOME=""
CONFIG_HOME=""
HF_TOKEN=${HF_TOKEN:-}

GRANITE_7B_MODEL="instructlab/granite-7b-lab"
MIXTRAL_8X7B_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1"

init_e2e_tests() {
    E2E_TEST_DIR=$(mktemp -d)
    export HOME="${E2E_TEST_DIR}"  # update the HOME directory used to resolve paths

    CONFIG_HOME=$(python -c 'import platformdirs; print(platformdirs.user_config_dir())')
    DATA_HOME=$(python -c 'import platformdirs; print(platformdirs.user_data_dir())')
    CACHE_HOME=$(python -c 'import platformdirs; print(platformdirs.user_cache_dir())')
    # ensure that our mock e2e dirs exist
    for dir in "${CONFIG_HOME}" "${DATA_HOME}" "${CACHE_HOME}"; do
        mkdir -p "${dir}"
    done

    E2E_LOG_DIR="${HOME}/log"
    mkdir -p "${E2E_LOG_DIR}"
}

test_train() {
    task initialize ilab
    # TODO: get profiles
    ilab config init --non-interactive --profile="${SCRIPTDIR}/test-data/profile-l40s-x4.yaml"
    mkdir -p "$DATA_HOME"/instructlab/taxonomy/knowledge/phoenix/overview/e2e-phoenix
    cp "$SCRIPTDIR"/test-data/e2e-qna-knowledge-phoenix.yaml "$DATA_HOME"/instructlab/taxonomy/knowledge/phoenix/overview/e2e-phoenix/qna.yaml
    
    mkdir -p "$DATA_HOME"/instructlab/taxonomy/compositional_skills/extraction/answerability/e2e-yes_or_no
    cp "$SCRIPTDIR"/test-data/e2e-qna-grounded-employee-skill.yaml "$DATA_HOME"/instructlab/taxonomy/compositional_skills/extraction/answerability/e2e-yes_or_no/qna.yaml
    task ilab initialization complete

    task download models

    step Downloading the mixtral-8x7b instruct model as the teacher model for SDG
    ilab model download --repository ${MIXTRAL_8X7B_MODEL} --hf-token "${HF_TOKEN}"
    step Downloading granite-7b-lab model to train
    ilab model download --repository ${GRANITE_7B_MODEL}
    
    task model downloading complete

    task generate ilab data
    ilab data generate --enable-serving-output
    task generation complete

    task Train the model with instructlab/training SDK

    local knowledge_data_path
    local skills_data_path
    knowledge_data_path=$(find "${DATA_HOME}"/instructlab/datasets -name 'knowledge_train_msgs*' | head -n 1)
    skills_data_path=$(find "${DATA_HOME}"/instructlab/datasets -name 'skills_train_msgs*' | head -n 1)


    export INSTRUCTLAB_EVAL_FIRST_N_QUESTIONS=10
    export HF_DATASETS_TRUST_REMOTE_CODE=true

    python -c "import sys; sys.path.insert(0, '${SCRIPTDIR}'); from test_sdk import run_test; run_test('${knowledge_data_path}', '${skills_data_path}', nnodes=1, node_rank=0, nproc_per_node=4)"
    task Training complete
}

step() {
    echo -e "$BOLD$* - $(date)$NC"
}

task() {
    echo -e "$BOLD------------------------------------------------------$NC"
    step "$@"
}

check_disk() {
   task Check disk
   df -h
}

init_e2e_tests
test_train