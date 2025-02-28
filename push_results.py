import os, sys
from datetime import datetime

import subprocess

#TODO REMOVE duplicate with pull_requets.py
def git_clone_or_pull(repo_url, folder):
    if not os.path.isdir(folder):
        try:
            # Run git clone command
            subprocess.run(["git", "clone", repo_url, folder], check=True)
            print(f"Repository cloned successfully to {folder}")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
    else :
        try:
            subprocess.run(["git", "-C", folder, "pull"], check=True)
            print(f"Repository pulled successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error pulling repository: {e}")

def git_commit_push(modified_repo):
    # Check if repo exists
    if not os.path.exists(modified_repo):
        print(f"Error: Repo '{modified_repo}' does not exist.")
        return

    try:
        # Add all files in the source directory to the repository
        subprocess.run(["git", "-C", modified_repo, "add", "."], check=True)

        print("All changes staged successfully.")
        # Commit the changes with a meaningful message including timestamp
        commit_message = f"Update results for leaderboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        subprocess.run(["git", "-C", modified_repo, "commit", "-m", commit_message], check=True)

        print("Changes committed successfully.")

        # Push the changes to the remote repository
        subprocess.run(["git", "-C", modified_repo, "push", "origin", "main"], check=True)  # Assuming the default branch is 'main'
        print("Changes pushed to remote repository successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Git operation failed with error code {e.returncode}")
        print(e.stderr)

def copy_result_files(source_dir, target_dir):
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

def push(results_dir):
    # Define environment variables
    hf_user = os.environ.get("HF_USER_ACCESS_GIT")
    hf_token = os.environ.get("HF_TOKEN_ACCESS_GIT")

    if not hf_user or not hf_token:
        print("Error: HF_USER_ACCESS_GIT and HF_TOKEN_ACCESS_GIT must be set in the environment.")
        sys.exit()

    # Construct repository URL TODO Update for fr-gouv-coordination-ia when going production
    repo_url = f"https://{hf_user}:{hf_token}@huggingface.co/datasets/fr-gouv-coordination-ia/results-dev"

    git_clone_or_pull(repo_url, "results")

    copy_result_files(results_dir, "results")

    # TODO do the same with requests and update requests status
    # Parse results to match requests and ensure the request status is OK

    git_commit_push("results")

if __name__ == "__main__":
    print("Trying to push results")
    push(sys.argv[1])