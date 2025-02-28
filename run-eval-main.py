import subprocess
import os
from pathlib import Path

def setup_environment():
    """
    Set up the environment variables and ensure apptainer is available.
    """

    subprocess.run("source /etc/bash.bashrc.g5k", shell=True, executable="/bin/bash")
    subprocess.run("module", shell=True, executable="/bin/bash")

    subprocess.run(
        "source /etc/profile.d/lmod.sh && module load apptainer",
        shell=True,
        executable="/bin/bash")

    return os.environ.copy()

def main():
    env = setup_environment()

    # Apptainer command arguments
    container_path = os.path.expanduser("~/llm_benchmark_fr.sif")
    bind_dirs = [
        f"--bind={os.getcwd()}",
        "--bind=/var/lib/oar",
        f"--bind={os.path.expanduser('~/.hf_token')}",
        f"--bind={os.path.expanduser('~/.ssh')}",
        f"--bind={os.path.expanduser('~/clearml.conf')}"
    ]

    # Run apptainer command
    subprocess.run(
        ["apptainer", "exec"] + bind_dirs + [container_path, "bash", "-c"],
        input='''\
echo "inside container"
echo $PWD
export TMP__DIR=/tmp/${USER}-runtime-dir
mkdir -p ${TMP__DIR}
export HF_HOME=${TMP__DIR}/cache
mkdir -p ${HF_OME}
export TRITON_ CACHE_ DIR=${TMP_ DIR}/triton
mkdir -p ${TRITON_ CACHE_ DIR}
export VLLM_ CONFIG_ ROOT=${TMP_ DIR}/.config/vllm
mkdir -p ${VLLM_ CONFIG_ ROOT}
HF_TOKEN=$(cat ~/.hf_token)
huggingface- cli login --token ${HF_TOKEN}
ray start --head --port=6379
NODES=$(cat ${OAR_FILE_NODES} | uniq | grep -v ${HOSTNAME})
echo "Other nodes: ${NODES}"
for host in $NODES; do
  ssh $host "bash -s" < run-eval-workers.sh $(hostname -I | cut -d " " -f1)
done
ray status
pip list
python3 run-lighteval.py''',
        env=env,
        shell=True,
        check=True,
        executable="/bin/bash"
    )

if __name__ == "__main__":
    main()
