#!/bin/bash

module load apptainer

apptainer exec --no-mount home --nv --bind $PWD --bind /var/lib/oar  --bind ~/.hf_token --bind ~/.hf_push_user --bind ~/.hf_push_token  --bind ~/.ssh --bind ~/clearml.conf ~/llm_benchmark_fr.sif bash -c '

echo "inside container"
echo $PWD

export TMP_DIR=/tmp/${USER}-runtime-dir
mkdir -p ${TMP_DIR}
export HF_HOME=${TMP_DIR}/cache
mkdir -p ${HF_HOME}
export TRITON_CACHE_DIR=${TMP_DIR}/triton
mkdir -p ${TRITON_CACHE_DIR}
export VLLM_CONFIG_ROOT=${TMP_DIR}/.config/vllm
mkdir -p ${VLLM_CONFIG_ROOT}
export RESULT_DIR=$TMP_DIR/results
export DETAIL_DIR=$RESULT_DIR/details

HF_TOKEN=$(cat ~/.hf_token)

huggingface-cli login --token ${HF_TOKEN}

NODES=($(cat ${OAR_FILE_NODES} | uniq | grep -v ${HOSTNAME}))
NNODES=${#NODES[@]}
echo "Number of nodes: $(($NNODES+1))"
echo "Current node: ${HOSTNAME}"

if (($NNODES>0)); then
   ray start --head --port=6379
   for ((i=0; i<${NNODES}; i++));
   do
	   echo "Other nodes: Index $i - Node ${NODES[i]}"
	   ssh ${NODES[i]} "bash -s" < run-eval-workers.sh $(hostname -I | cut -d " " -f1)
   done
   ray status
else
   export VLLM_WORKER_MULTIPROC_METHOD=spawn
fi

pip list

## Dynamic number of gpu per node and total gpus
#NGPUSPERNODES=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
#NGPUS=$(($NGPUSPERNODES*($NNODES+1)))

python3 run-lighteval.py

export HF_USER_ACCESS_GIT=$(cat ~/.hf_push_user)
export HF_TOKEN_ACCESS_GIT=$(cat ~/.hf_push_token)
python3 push_results.py $TMP_DIR/results
'
