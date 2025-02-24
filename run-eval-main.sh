#!/bin/bash

module load apptainer

apptainer exec --no-mount home --nv --bind $PWD --bind /var/lib/oar  --bind ~/.hf_token --bind ~/.ssh --bind ~/clearml.conf ~/llm_benchmark_fr.sif bash -c '

echo "inside container"
echo $PWD

export HF_HOME=/tmp/${USER}-runtime-dir/cache
mkdir -p ${HF_HOME}
export TRITON_CACHE_DIR=/tmp/${USER}-runtime-dir/triton
mkdir -p ${TRITON_CACHE_DIR}
export VLLM_CONFIG_ROOT=/tmp/${USER}-runtime-dir/.config/vllm
mkdir -p ${VLLM_CONFIG_ROOT}

HF_TOKEN=$(cat ~/.hf_token)

huggingface-cli login --token ${HF_TOKEN}

ray start --head --port=6379

NODES=$(cat ${OAR_FILE_NODES} | uniq | grep -v ${HOSTNAME})
echo "Other nodes: ${NODES}"
for host in $NODES;
do
  scp run-eval-workers.sh $host:/tmp/${USER}/run-eval-workers.sh
  ssh $host /tmp/${USER}:run-eval-workers.sh $(hostname -I | cut -d " " -f1)
done

ray status
pip list

python3 run-lighteval.py
'
