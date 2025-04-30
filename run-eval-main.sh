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

NODES=($(cat ${OAR_FILE_NODES} | uniq | grep -v ${HOSTNAME}))
NNODES=${#NODES[@]}
echo "Number of nodes: $(($NNODES+1))"
echo "Current node: ${HOSTNAME}"

if (($NNODES>0)); then
   if (($NNODES>2)); then
      export RAY_CGRAPH_submit_timeout=100
      export RAY_CGRAPH_get_timeout=100
   fi

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
if [ -d "$OUTPUT_DIR/results" ]; then
  mv $OUTPUT_DIR/results $RESULT_DIR

  ## The following is a hacky way of retrieving the folder
  ## containing the prediction parquet files, in order to
  ## add the log there.
  ## It makes the assumption that there is alway a subdirectory
  ## structure like details/provider/model/datetime/
  PARQUET_DETAIL_DIR=$DETAIL_DIR/*/*/

  echo "Resolved PARQUET_DETAIL_DIR: $(realpath $PARQUET_DETAIL_DIR)"

  STDERR_FILE="$HOME/OAR.LLM evaluation.${OAR_JOB_ID}.stderr"
  STDOUT_FILE="$HOME/OAR.LLM evaluation.${OAR_JOB_ID}.stdout"

  for FILE in "$STDERR_FILE" "$STDOUT_FILE"; do
    if [ -f "$FILE" ]; then
       cp "$FILE" "$PARQUET_DETAIL_DIR"
    else
      echo "Error: '$FILE' not found." >&2
    fi
  done

  export HF_USER_ACCESS_GIT=$(cat ~/.hf_push_user)
  export HF_TOKEN_ACCESS_GIT=$(cat ~/.hf_push_token)
  python3 push_results.py $DETAIL_DIR "details-dev"
  rm -rf $DETAIL_DIR
  python3 push_results.py $RESULT_DIR "results-dev"

else
  echo "No $OUTPUT_DIR/results directory, error"
  exit 1
fi
