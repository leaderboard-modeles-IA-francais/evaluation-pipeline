#!/bin/bash
module load apptainer

apptainer exec --no-mount home,cwd --bind ${HOME}/.hf_token --env ADD=$1 --nv ~/llm_benchmark_fr.sif bash -c '

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
