import os, sys
from datetime import datetime
import subprocess
from git_utils import git_clone_or_pull, git_commit_push

def copy_result_files(source_dir, target_dir):
    print(f"Trying to copy {source_dir} into {target_dir}")

    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Directory '{source_dir}' does not exist.")
        return

    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    try:
        subprocess.run(["cp", "-r", f'{source_dir}', f'{target_dir}'], check=True)
        print("All results copied to git directory.")

    except subprocess.CalledProcessError as e:
        print(f"Files copy failed with error code {e.returncode}")
        print(e.stderr)

def push(results_dir, results_repo):
    # Define environment variables
    hf_user = os.environ.get("HF_USER_ACCESS_GIT")
    hf_token = os.environ.get("HF_TOKEN_ACCESS_GIT")

    if not hf_user or not hf_token:
        print("Error: HF_USER_ACCESS_GIT and HF_TOKEN_ACCESS_GIT must be set in the environment.")
        sys.exit()

    # Construct repository URL TODO Update for fr-gouv-coordination-ia when going production
    repo_url = f"https://{hf_user}:{hf_token}@huggingface.co/datasets/fr-gouv-coordination-ia/" + results_repo

    git_clone_or_pull(repo_url, results_repo)

    copy_result_files(results_dir, results_repo)

    # TODO do the same with requests and update requests status
    # Parse results to match requests and ensure the request status is OK

    git_commit_push(results_repo)

def filter_model_with_results(models, results_repo):
    """
    Return only models that do NOT already have results in results_dir/results_repo.
    Assumes each model's result is a directory or file named after the model (or a normalized version).
    """
    # Define environment variables
    hf_user = os.environ.get("HF_USER_ACCESS_GIT")
    hf_token = os.environ.get("HF_TOKEN_ACCESS_GIT")

    if not hf_user or not hf_token:
        print("Error: HF_USER_ACCESS_GIT and HF_TOKEN_ACCESS_GIT must be set in the environment.")
        sys.exit()

    # Construct repository URL TODO Update for fr-gouv-coordination-ia when going production
    repo_url = f"https://{hf_user}:{hf_token}@huggingface.co/datasets/fr-gouv-coordination-ia/" + results_repo

    git_clone_or_pull(repo_url, results_repo)

    filtered_models = []
    for model in models:
        if not os.path.exists(os.path.join(results_repo, model, "results")):
            filtered_models.append(model)
    return filtered_models
