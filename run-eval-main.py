import subprocess
import os
from pathlib import Path

def setup_environment():
    """
    Set up the environment variables and ensure apptainer is available.
    """

    subprocess.run(
        "source /etc/profile.d/lmod.sh && env",
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True
    )

    # Check if 'module' command exists and try to load apptainer
    try:
        subprocess.run(["module", "load", "apptainer"], check=True)
        print("Successfully loaded apptainer module.")
    except FileNotFoundError:
        # If 'module' is not found, assume apptainer is in the PATH or set the path explicitly
        print("Module command not found. Attempting to use apptainer directly.")

    # Verify if apptainer is available
    try:
        subprocess.run(["which", "apptainer"], check=True)
        print("Apptainer executable found.")
    except subprocess.CalledProcessError:
        # If apptainer is not in PATH, provide an alternative path (e.g., where you installed it)
        apptainer_path = "/path/to/your/apptainer"  # Replace with the actual path
        os.environ["PATH"] += f":{apptainer_path}"
        print(f"Added apptainer to PATH: {apptainer_path}")

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
        check=True
    )

if __name__ == "__main__":
    main()
