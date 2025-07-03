from clearml import PipelineController, Task

import os
import math
import subprocess
import requests

import pull_requests
import push_results

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

model_nb_gpus_per_node_map = {

}

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
    "EpistemeAI/ReasoningCore-Llama-3.2-3B-R01-1.1": "24:00",
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

pipe = PipelineController(
  name="LLM Leaderboard FR Pipeline on musa", project=project_name, version="1.0.0"
  #continue_on_fail=True,
  #continue_on_abort=True,
  #skip_children_on_fail=False,
  #skip_chlidren_on_abort=False
)

eval_tasks = [ ]

# Retrieve all models which need to be evaluated
models = pull_requests.models()

models = list(set(models))

for model in models:
    if model in model_too_large_list:
        continue

    if model in model_incompatible_list:
        continue

    hf_token = os.environ.get("HF_TOKEN_ACCESS_MODELS")
    if not hf_token:
        print("Error: HF_TOKEN_ACCESS_MODELS must be set in the environment.")
        sys.exit()

    # Retrieve the configuration of each model
    try:
        # Use requests instead of subprocess for better error handling
        response = requests.get(
            f'https://huggingface.co/{model}/resolve/main/config.json',
            headers={'Authorization': f'Bearer {hf_token}'}
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching config file: {e}")
        continue

    # Extract the JSON content
    try:
        config = response.json()
    except json.JSONDecodeError:
        print("Failed to parse config.json")
        exit(1)

    nb_nodes = model_nb_nodes_map.get(model, 2)
    nb_gpus_per_node = model_nb_gpus_per_node_map.get(model, 2)
    gpu_memory_utilization = model_gpu_memory_utilization_map.get(model, 0.5)
    walltime = model_walltime_map.get(model, "02:00")

    task_name = f"eval_{model}"
    eval_tasks.append(task_name)
    pipe.add_step(
        name=task_name,
        parents=[ ],
        base_task_project=project_name,
        base_task_name='eval_model',
        parameter_override={
            #'General/dataset_url': '${stage_data.artifacts.dataset.url}',
            'General/model': model,
            'General/cluster': 'musa',
            'General/nb_nodes': nb_nodes,
            'General/nb_gpus_per_node': nb_gpus_per_node,
            'General/gpu_memory_utilization': gpu_memory_utilization,
            'General/tasks': 'community|bac-fr|0|0,community|ifeval-fr|0|0,community|gpqa-fr|0|0',
            'General/max_model_length': None,
            'General/use_chat_template': True,
            'General/walltime': walltime
        },
        execution_queue='national_clusters',
        #pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback
    )

pipe.start_locally()
