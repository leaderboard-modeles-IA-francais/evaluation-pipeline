import os
import subprocess
from subprocess import run, CalledProcessError
from datetime import datetime

def git_clone_or_pull(repo_url, folder):
    if not os.path.isdir(folder):
        try:
            # Run git clone command
            try:
                run(["git", "clone", repo_url, folder], check=True)
            except NameError:
                subprocess.run(["git", "clone", repo_url, folder], check=True)
            print(f"Repository cloned successfully to {folder}")
        except Exception as e:
            print(f"Error cloning repository: {e}")
    else:
        try:
            try:
                run(["git", "-C", folder, "pull"], check=True)
            except NameError:
                subprocess.run(["git", "-C", folder, "pull"], check=True)
            print(f"Repository pulled successfully")
        except Exception as e:
            print(f"Error pulling repository: {e}")

def git_commit_push(modified_repo):
    # Check if repo exists
    if not os.path.exists(modified_repo):
        print(f"Error: Repo '{modified_repo}' does not exist.")
        return
    try:
        # Add all files in the source directory to the repository
        try:
            run(["git", "-C", modified_repo, "add", "."], check=True)
        except NameError:
            subprocess.run(["git", "-C", modified_repo, "add", "."], check=True)
        print("All changes staged successfully.")
        # Commit the changes with a meaningful message including timestamp
        commit_message = f"Update results for leaderboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        try:
            run(["git", "-C", modified_repo, "commit", "-m", commit_message], check=True)
        except NameError:
            subprocess.run(["git", "-C", modified_repo, "commit", "-m", commit_message], check=True)
        print("Changes committed successfully.")
        # Push the changes to the remote repository
        try:
            run(["git", "-C", modified_repo, "push", "origin", "main"], check=True)
        except NameError:
            subprocess.run(["git", "-C", modified_repo, "push", "origin", "main"], check=True)
        print("Changes pushed to remote repository successfully.")
    except Exception as e:
        print(f"Git operation failed: {e}") 