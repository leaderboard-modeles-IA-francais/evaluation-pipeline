#!/bin/bash

module load apptainer

apptainer exec --no-mount home --nv --bind $PWD --bind /var/lib/oar  --bind ~/.hf_token --bind ~/.ssh --bind ~/clearml.conf ~/llm_benchmark_fr.sif bash -c '

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

HF_TOKEN=$(cat ~/.hf_token)

huggingface-cli login --token ${HF_TOKEN}

NODES=($(cat ${OAR_FILE_NODES} | uniq | grep -v ${HOSTNAME}))
NNODES=${#NODES[@]}
echo "Number of nodes: $(($NNODES+1))"
echo "Current node: ${HOSTNAME}"

ray start --head --port=6379

if (($NNODES>0)); then
   for ((i=0; i<${NNODES}; i++));
   do
	echo "Other nodes: Index $i - Node ${NODES[i]}"
	ssh ${NODES[i]} "bash -s" < run-eval-workers.sh $(hostname -I | cut -d " " -f1)
   done

else
   export VLLM_WORKER_MULTIPROC_METHOD=spawn
fi

ray status
pip list

NGPUSPERNODES=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NGPUS=$(($NGPUSPERNODES*($NNODES+1)))

python3 run-lighteval.py
'
