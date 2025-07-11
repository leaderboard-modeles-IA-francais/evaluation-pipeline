from clearml import PipelineController, Task
import os, sys
import math
import subprocess
import requests
import json
import git_requests
import git_results

results = {}

project_name = "LLM Leaderboard FR"

model_too_large_list = ["allenai/Llama-3.1-Tulu-3-405B"]
model_incompatible_list = ["teapotai/teapotllm"]
model_nb_nodes_map = {
    "mistralai/Mistral-Large-Instruct-2411": 4,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": 4,
    "eval_mistralai/Mixtral-8x22B-Instruct-v0.1": 4,
    "Qwen/Qwen2.5-Math-72B-Instruct": 4,
    "microsoft/phi-4": 1,
    "microsoft/Phi-3-medium-128k-instruct": 1,
    "jpacifico/Chocolatine-2-14B-Instruct-v2.0": 1,
    "EpistemeAI/ReasoningCore-Llama-3.2-3B-R01-1.1": 1,
}
model_gpu_memory_utilization_map = {
    "mistralai/Mistral-Large-Instruct-2411": 0.8,
}
model_nb_gpus_per_node_map = {}
model_walltime_map = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "04:30",
    "Qwen/Qwen2.5-Math-72B-Instruct": "14:00",
    "mistralai/Mistral-Large-Instruct-2411": "08:00",
    "EpistemeAI/ReasoningCore-Llama-3.2-3B-R01-1.1": "06:00",
    "speakleash/Bielik-11B-v2.3-Instruct": "05:00",
    "HoangHa/Pensez-v0.1-e5": "03:00",
    "MaziyarPanahi/calme-3.2-instruct-78b": "03:00",
    "arcee-ai/Virtuoso-Medium-v2": "12:00",
    "HoangHa/Pensez-Llama3.1-8B": "03:00",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "03:00",
    "baconnier/Napoleon_24B_V0.1": "08:00",
}
continuation_behaviour = {
    "continue_on_fail": True,
    "continue_on_abort": True,
    "skip_children_on_fail": False,
    "skip_children_on_abort": False,
}

def print_results():
    print(f"Model name | IFEVAL-FR | GPQA-FR | BAC-FR")
    print( "-----------------------------------------------------")
    for m,r in results.items():
        print(f"{m} | {r['ifeval-fr']} | {r['gpqa-fr']} | {r['bac-fr']}")

def post_execute_callback(a_pipeline, a_node):
    print("Completed Task id={}".format(a_node.executed))
    if a_node.executed:
        completed = Task.get_task(task_id=a_node.executed)
        model_name = completed.get_parameter('General/model')
        metrics = completed.get_reported_single_values()
        results[model_name] = {}
        results[model_name]['ifeval-fr'] = (metrics['community:ifeval-fr:0 | prompt_level_strict_acc'] + metrics['community:ifeval-fr:0 | inst_level_strict_acc']) / 2 * 100
        results[model_name]['gpqa-fr'] = metrics['community:gpqa-fr:0 | acc'] * 100
        results[model_name]['bac-fr'] = metrics['community:bac-fr:0 | bac-fr-qem'] * 100
        print_results()
    return

def create_model_list():
    """Create the list of models to evaluate"""
    models = git_requests.pending_models()
    models = list(set(models))  # Remove duplicates
    
    valid_models = []
    for model in models:
        if model in model_too_large_list:
            print(f"Skipping {model} - too large")
            continue
        if model in model_incompatible_list:
            print(f"Skipping {model} - incompatible")
            continue
        
        hf_token = os.environ.get("HF_TOKEN_ACCESS_MODELS")
        if not hf_token:
            print("Error: HF_TOKEN_ACCESS_MODELS must be set in the environment.")
            sys.exit()
        
        try:
            response = requests.get(
                f'https://huggingface.co/{model}/resolve/main/config.json',
                headers={'Authorization': f'Bearer {hf_token}'}
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Error fetching config file for {model}: {e}")
            continue
        
        try:
            config = response.json()
        except json.JSONDecodeError:
            print(f"Failed to parse config.json for {model}")
            continue
        
        valid_models.append(model)
        print(f"Added {model} to evaluation queue")
    
    return valid_models

# Create the pipeline
pipe = PipelineController(
    name="LLM Leaderboard FR Pipeline on chuc", 
    project=project_name, 
    version="1.0.0"
)

# Get the list of models to evaluate
models_to_evaluate = create_model_list()
print(f"Total models to evaluate: {len(models_to_evaluate)}")

# Add evaluation tasks sequentially
prev_task_name = None
for model in models_to_evaluate:
    nb_nodes = model_nb_nodes_map.get(model, 2)
    nb_gpus_per_node = model_nb_gpus_per_node_map.get(model, 4)
    gpu_memory_utilization = model_gpu_memory_utilization_map.get(model, 0.5)
    walltime = model_walltime_map.get(model, "02:00")
    
    task_name = f"eval_{model}"
    pipe.add_step(
        name=task_name,
        parents=[prev_task_name] if prev_task_name else [],
        base_task_project=project_name,
        base_task_name='eval_model',
        parameter_override={
            'General/model': model,
            'General/cluster': 'chuc',
            'General/nb_nodes': nb_nodes,
            'General/nb_gpus_per_node': nb_gpus_per_node,
            'General/gpu_memory_utilization': gpu_memory_utilization,
            'General/tasks': 'community|bac-fr|0|0,community|ifeval-fr|0|0,community|gpqa-fr|0|0',
            'General/max_model_length': None,
            'General/use_chat_template': True,
            'General/walltime': walltime
        },
        execution_queue='national_clusters',
        post_execute_callback=post_execute_callback
    )
    prev_task_name = task_name

pipe.start()
