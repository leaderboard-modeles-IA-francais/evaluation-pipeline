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


def post_execute_callback(a_pipeline, a_node):
    print("Completed Task id={}".format(a_node.executed))
    return

pipe = PipelineController(
  name="LLM Leaderboard FR Pipeline", project=project_name, version="1.0.1"
  #continue_on_fail=True,
  #continue_on_abort=True,
  #skip_children_on_fail=False,
  #skip_chlidren_on_abort=False
)

pipe.add_parameter(
    name='cluster',
    description='Cluster on which the tasks will be executed. Current available options are "musa", "chuc", "auto".',
    default='musa'
)

pipe.add_parameter(
    name='tasks',
    description='Lighteval tasks to execute.',
    default='community|bac-fr|0|0,community|ifeval-fr|0|0,community|pr-fouras|0|0,community|gpqa-fr|0|0'
)

pipe.add_parameter(
    name='use_chat_template',
    description='Whether to use chat templates or not.',
    default=True
)

eval_tasks = [ ]

# Retrieve all models which need to be evaluated
models = pull_requests.models()

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

    # TODO: Check entries are present or fail
    # Get the number of attention heads (ensure it's an integer)
    num_attention_heads = int(config.get("num_attention_heads", 0))

    # The max number of tokens of the model is not retrieved nor passed to the task since it is retrieved by lighteval and / or vllm from the model itself.

    # Compute number of nodes for musa
    # Get number of gpus to use (6 * 2 gpus)
    nb_available_gpus = 4 # TODO: For testing purpose
    for j in range(min(nb_available_gpus, num_attention_heads), 1, -1):
        if num_attention_heads % j == 0 :
            nb_gpus = j
            break

    # Get number of nodes to use
    nb_gpus_per_node = 2
    nb_nodes = math.ceil(nb_gpus / nb_gpus_per_node)

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
            'General/cluster': '${pipeline.cluster}',
            'General/nb_nodes': nb_nodes,
            'General/nb_gpus_per_node': nb_gpus_per_node,
            'General/gpu_memory_utilization': 0.5,
            'General/tasks': '${pipeline.tasks}',
            'General/max_model_length': None,
            'General/use_chat_template': '${pipeline.use_chat_template}'
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
