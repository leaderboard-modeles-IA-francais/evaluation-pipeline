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
    
    # Initialize ClearML task
    task = Task.init(project_name="LLM Leaderboard FR", task_name="eval_model_inspect_ai")
    
    # Default parameters matching lighteval setup
    parameters = {
        'model': 'meta-llama/Llama-3.2-3B-Instruct',
        'dtype': 'bfloat16', 
        'gpu_memory_utilization': 0.5,
        'nb_gpus_per_node': 4,
        'nb_nodes': 1,
        'enforce_eager': True,
        'tasks': 'community|bac-fr|0|0,community|pr-fouras|0|0,community|gpqa-fr|0|0',
        'max_model_length': None,
        'use_chat_template': True,
        'framework': 'inspect_ai'  # New parameter to distinguish framework
    }
    
    task.connect(parameters)
    
    # Check chat template
    use_chat_template = has_chat_template(parameters['model'])
    print(f"Model has chat_template? {use_chat_template}")
    task.set_parameter("use_chat_template", use_chat_template)
    
    # Parse tasks
    task_names = parse_tasks_string(parameters['tasks'])
    print(f"Running inspect_ai evaluation on tasks: {task_names}")
    
    if not task_names:
        print("No valid tasks found in task string")
        return
    
    # Create model configuration for inspect_ai
    # Note: inspect_ai uses different model backends than vLLM
    # We'll use the model name directly and let inspect_ai handle the backend
    model_name = parameters['model']
    
    # For inspect_ai, we can use different model providers
    # Common options: "openai/model", "anthropic/model", "local/model"
    # For now, we'll use a generic approach that works with transformers
    try:
        # Try using transformers backend for local models
        model = get_model(f"hf/{model_name}")
    except:
        try:
            # Fallback to generic model specification
            model = get_model(model_name)
        except Exception as e:
            print(f"Warning: Could not create model {model_name}: {e}")
            print("Using mock model for testing")
            model = None
    
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
            
            # Skip evaluation if model is None (for testing)
            if model is None:
                print(f"Skipping {task_name} - no model available")
                results[task_name] = {"error": "No model available"}
                continue
            
            # Run evaluation
            eval_result = eval(
                task_obj,
                model=model,
                log_dir=output_dir
            )
            
            # Extract results - handle different result formats
            if hasattr(eval_result, 'results'):
                if hasattr(eval_result.results, 'scores'):
                    scores = eval_result.results.scores
                else:
                    scores = {}
                
                if hasattr(eval_result.results, 'metrics'):
                    metrics = eval_result.results.metrics
                else:
                    metrics = {}
            else:
                # Fallback for different result formats
                scores = {}
                metrics = getattr(eval_result, 'metrics', {})
            
            results[task_name] = {
                "scores": scores,
                "metrics": metrics,
                "status": "completed"
            }
            
            print(f"Task {task_name} completed successfully")
            
        except Exception as e:
            print(f"Error running task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            results[task_name] = {"error": str(e), "status": "failed"}
    
    # Save consolidated results
    results_file = Path(output_dir) / "inspect_ai_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log results to ClearML
    logger = task.get_logger()
    
    for task_name, task_results in results.items():
        if "error" not in task_results:
            for metric_name, value in task_results.get("metrics", {}).items():
                logger.report_single_value(f"{task_name} | {metric_name}", value)
        else:
            print(f"Task {task_name} failed: {task_results['error']}")
    
    # Upload results as artifact
    task.upload_artifact(name='inspect_ai_results', artifact_object=str(results_file))
    
    print("\nInspect AI evaluation completed!")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()