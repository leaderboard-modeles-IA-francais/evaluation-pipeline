import os
import json
from pathlib import Path
from clearml import Task
from transformers import AutoTokenizer

# Import inspect_ai components
from inspect_ai import eval
from inspect_ai.model import get_model
from inspect_ai.log import read_eval_log

# Import our custom tasks
from tasks.french_evals_inspect import AVAILABLE_TASKS


def has_chat_template(model):
    """Check if model has chat template"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        return getattr(tokenizer, "chat_template", None) is not None
    except Exception:
        return False


def parse_tasks_string(tasks_str: str) -> list:
    """Parse the tasks string format used by lighteval"""
    if not tasks_str:
        return []
    
    tasks = []
    for task_spec in tasks_str.split(','):
        # Format: community|task-name|shots|batch_size
        parts = task_spec.strip().split('|')
        if len(parts) >= 2:
            task_name = parts[1]
            if task_name in AVAILABLE_TASKS:
                tasks.append(task_name)
    
    return tasks


def main():
    unix_user = os.environ.get("USER")
    output_dir = f"/tmp/{unix_user}-runtime-dir/results"
    
    # For interactive mode, don't init ClearML task automatically
    print("Running inspect_ai in interactive mode")
    
    # Interactive parameters - can be modified for testing
    parameters = {
        'model': 'meta-llama/Llama-3.2-3B-Instruct',
        'tasks': 'community|bac-fr|0|0,community|pr-fouras|0|0',  # Fewer tasks for testing
        'framework': 'inspect_ai'
    }
    
    # Check chat template
    use_chat_template = has_chat_template(parameters['model'])
    print(f"Model has chat_template? {use_chat_template}")
    
    # Parse tasks
    task_names = parse_tasks_string(parameters['tasks'])
    print(f"Running inspect_ai evaluation on tasks: {task_names}")
    
    if not task_names:
        print("No valid tasks found in task string")
        return
    
    # Create model configuration for inspect_ai
    model_name = parameters['model']
    
    # For testing, use a simpler model configuration
    model = get_model(f"openai/{model_name}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Run each task
    for task_name in task_names:
        print(f"\nRunning task: {task_name}")
        
        try:
            # Get the task function
            task_fn = AVAILABLE_TASKS[task_name]
            
            # Create the task
            task_obj = task_fn()
            
            # Run evaluation with limited samples for testing
            eval_result = eval(
                task_obj,
                model=model,
                log_dir=output_dir,
                limit=10  # Limit samples for interactive testing
            )
            
            # Extract results
            results[task_name] = {
                "scores": eval_result.results.scores,
                "metrics": eval_result.results.metrics
            }
            
            print(f"Task {task_name} completed successfully")
            
        except Exception as e:
            print(f"Error running task {task_name}: {e}")
            results[task_name] = {"error": str(e)}
    
    # Save consolidated results
    results_file = Path(output_dir) / "inspect_ai_results_interactive.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nInspect AI interactive evaluation completed!")
    print(f"Results saved to: {results_file}")
    
    # Print summary
    for task_name, task_results in results.items():
        if "error" not in task_results:
            print(f"\n{task_name} Results:")
            for metric_name, value in task_results.get("metrics", {}).items():
                print(f"  {metric_name}: {value}")
        else:
            print(f"\n{task_name} FAILED: {task_results['error']}")


if __name__ == "__main__":
    main()