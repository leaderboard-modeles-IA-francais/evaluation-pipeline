#!/bin/bash

module load conda
conda activate py3.12

echo "inside conda env"
echo $PWD
echo $CONDA_PREFIX

export TMP_DIR=/tmp/${USER}-runtime-dir
mkdir -p ${TMP_DIR}
export HF_HOME=${TMP_DIR}/cache
mkdir -p ${HF_HOME}
export TRITON_CACHE_DIR=${TMP_DIR}/triton
mkdir -p ${TRITON_CACHE_DIR}
export VLLM_CONFIG_ROOT=${TMP_DIR}/.config/vllm
mkdir -p ${VLLM_CONFIG_ROOT}
export OUTPUT_DIR=$TMP_DIR/results
export DETAIL_DIR=$OUTPUT_DIR/details
#export RESULT_DIR=$OUTPUT_DIR/_tmpdir_${USER}
export RESULT_DIR=$OUTPUT_DIR/_tmpdir_wr_${USER}


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
   echo " /!\ Mononoeud setup but those env variables that were needed are now problematic /!\ "
   # export VLLM_WORKER_MULTIPROC_METHOD=spawn
   # export VLLM_SKIP_P2P_CHECK=1
fi

#pip list

python3 run-lighteval-interactive.py
rsync -avzh --remove-source-files $OUTPUT_DIR/results/* $RESULT_DIR

export HF_USER_ACCESS_GIT=$(cat ~/.hf_push_user)
export HF_TOKEN_ACCESS_GIT=$(cat ~/.hf_push_token)
python3 push_results.py $RESULT_DIR
