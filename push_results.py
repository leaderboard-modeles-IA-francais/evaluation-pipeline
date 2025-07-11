import sys
import git_results

if __name__ == "__main__":
    results = sys.argv[1]
    results_repo = sys.argv[2]
    print(f"Trying to push {results} results to {results_repo}")
    git_results.push(results, results_repo) 