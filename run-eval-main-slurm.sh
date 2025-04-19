#!/bin/bash

echo $PWD

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
export RESULT_DIR=$OUTPUT_DIR/clearML-sprint1-wr

HF_TOKEN=$(cat ~/.hf_token)

huggingface-cli login --token ${HF_TOKEN}

NNODES=0
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
fi

pip list

## Dynamic number of gpu per node and total gpus
#NGPUSPERNODES=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
#NGPUS=$(($NGPUSPERNODES*($NNODES+1)))

python3 run-lighteval.py
rm -rf $DETAIL_DIR
if [ -d "$OUTPUT_DIR/results" ]; then
  mv $OUTPUT_DIR/results $RESULT_DIR

  export HF_USER_ACCESS_GIT=$(cat ~/.hf_push_user)
  export HF_TOKEN_ACCESS_GIT=$(cat ~/.hf_push_token)
  python3 push_results.py $RESULT_DIR
else
  echo "No $OUTPUT_DIR/results directory, error"
  exit 1
fi
