#!/bin/bash
module load apptainer

# Accept framework as second argument, default to lighteval
RAY_ADDRESS=$1
FRAMEWORK=${2:-lighteval}

if [ "$FRAMEWORK" = "inspect_ai" ]; then
    CONTAINER_NAME="llm_benchmark_fr_inspect.sif"
    echo "Using inspect_ai container: $CONTAINER_NAME"
else
    CONTAINER_NAME="llm_benchmark_fr.sif" 
    echo "Using lighteval container: $CONTAINER_NAME"
fi

apptainer exec --no-mount home,cwd --bind ${HOME}/.hf_token --env ADD=$RAY_ADDRESS --nv ~/$CONTAINER_NAME bash -c '

export HF_HOME=/tmp/${USER}-runtime-dir/cache
mkdir -p ${HF_HOME}
export TRITON_CACHE_DIR=/tmp/${USER}-runtime-dir/triton
mkdir -p ${TRITON_CACHE_DIR}
export VLLM_CONFIG_ROOT=/tmp/${USER}-runtime-dir/.config/vllm
mkdir -p ${VLLM_CONFIG_ROOT}

HF_TOKEN=$(cat ~/.hf_token)

huggingface-cli login --token ${HF_TOKEN}
ray start --address=$ADD":6379"
'
