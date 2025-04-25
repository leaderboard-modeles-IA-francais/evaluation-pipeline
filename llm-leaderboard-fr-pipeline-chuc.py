from clearml import PipelineController, Task

import os
import math
import subprocess
import requests

import pull_requests
import push_results

results = {}

project_name = "LLM Leaderboard FR"

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
        print(completed.get_registered_artifacts())
        print_results()
    return

pipe = PipelineController(
  name="LLM Leaderboard FR Pipeline on chuc", project=project_name, version="1.0.0"
  #continue_on_fail=True,
  #continue_on_abort=True,
  #skip_children_on_fail=False,
  #skip_chlidren_on_abort=False
)

eval_tasks = [ ]

# Retrieve all models which need to be evaluated
models = ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"]

for model in models:
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

    nb_nodes = 1
    nb_gpus_per_node = 4

    task_name = f"eval_{model}"
    eval_tasks.append(task_name)
    pipe.add_step(
        name=task_name,
        parents=[ ],
        base_task_project=project_name,
        base_task_name='eval_model_chuc',
        parameter_override={
            #'General/dataset_url': '${stage_data.artifacts.dataset.url}',
            'General/model': model,
            'General/cluster': 'chuc',
            'General/nb_nodes': str(nb_nodes),
            'General/nb_gpus_per_node': str(nb_gpus_per_node),
            'General/gpu_memory_utilization': 0.5,
            'General/tasks': 'community|bac-fr|0|0,community|ifeval-fr|0|0,community|gpqa-fr|0|0',
            'General/max_model_length': None,
            'General/use_chat_template': True
        },
        execution_queue='national_clusters',
        #pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback
    )

def gather_results():
    return True

pipe.add_function_step(
    name='gather_results',
    parents=eval_tasks,
    function=gather_results,
    function_return=[],
    execution_queue='default',
)

pipe.start_locally()
