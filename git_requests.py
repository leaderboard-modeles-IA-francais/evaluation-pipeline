import os, sys, requests, json
from subprocess import run, CalledProcessError
from datetime import datetime
from git_utils import git_clone_or_pull, git_commit_push

def parse_json_files(directory):
    """Parse all JSON files in the given directory (including subdirectories)"""
    json_data = {}

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        json_data[file_path] = data
                except Exception as e:
                    print(f"Error parsing {file_path}: {e}")

    return json_data

def filter_pending(json_data):
    """Filter JSON objects with status "PENDING" """
    pending_list = []

    for item in json_data.items():
        try:
#            if item[1].get('status') == 'PENDING':
            pending_list.append(item)
        except Exception as e:
            print(f"Error checking status for {item[0]}: {e}")

    return pending_list

def reorder_by_submitted_time(pending_items):
    """Reorder the list by submitted_time (oldest to most recent)"""
    try:
        # Sort by submitted_time in ascending order
        sorted_list = sorted(
            pending_items,
            key=lambda x: x[1].get('submitted_time', ''),  # Use empty string as default if field is missing
            reverse=False  # False for oldest to newest (ascending)
        )

        return sorted_list

    except Exception as e:
        print(f"Error reordering items: {e}")
        return pending_items

def reorder_by_likes(pending_items):
    """Reorder the list by likes (...)"""
    try:
        # Sort by submitted_time in ascending order
        sorted_list = sorted(
            pending_items,
            key=lambda x: x[1].get('likes', 0),
            reverse=True  # False for oldest to newest (ascending)
        )

        return sorted_list

    except Exception as e:
        print(f"Error reordering items: {e}")
        return pending_items

def write_updated_json(file_path, data):
    """Write updated JSON back to the file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)  # Save with indentation for readability
        print(f"Successfully updated {file_path}")
    except Exception as e:
        print(f"Error updating {file_path}: {e}")

def pending_models():
    # Define environment variables
    hf_user = os.environ.get("HF_USER_ACCESS_GIT")
    hf_token = os.environ.get("HF_TOKEN_ACCESS_GIT")

    if not hf_user or not hf_token:
        print("Error: HF_USER_ACCESS_GIT and HF_TOKEN_ACCESS_GIT must be set in the environment.")
        sys.exit()

    # Construct repository URL TODO Update for fr-gouv-coordination-ia when going production
    repo_url = f"https://{hf_user}:{hf_token}@huggingface.co/datasets/fr-gouv-coordination-ia/requests-dev"

    git_clone_or_pull(repo_url, "requests")

    directory_path = "./requests"

    os.chdir(directory_path)
    json_data = parse_json_files(os.getcwd())

    print("\nOriginal List:")
    for item in json_data:
        print(item)

    # Filter PENDING items
    pending_list = filter_pending(json_data)
    print("\nOriginal Pending List:")
    for item in pending_list:
        print(item)

    # # Reorder by submitted_time (from oldest to most recent) ----> Other strategy?
    # reordered_pending = reorder_by_submitted_time(pending_list)
    # print("\nReordered Pending List (Oldest to Most Recent):")
    # for item in reordered_pending:
    #     print(item)

    # Reorder by likes (from most likes to least likes)
    reordered_pending = reorder_by_likes(pending_list)
    print("\nReordered Pending List (Oldest to Most Recent):")
    for item in reordered_pending:
        print(item)

    # Update status to "EVALUATING" and write changes back to files
    if len(reordered_pending) == 0 :
        sys.exit()

    models = []
    for item in reordered_pending:
        models.append(item[1]["model"])

    return models 