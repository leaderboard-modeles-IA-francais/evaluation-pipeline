from clearml import PipelineController, Task

results = {}

def gather_results():
    return True

project_name = "LLM Leaderboard FR"

continuation_behaviour = {
    "continue_on_fail": True,
    "continue_on_abort": True,
    "skip_children_on_fail": False,
    "skip_children_on_abort": False,
}

def print_results():
    print(f"Model name | IFEVAL-FR | GPQA-FR | BAC-FR | PR-FOURAS")
    print( "-----------------------------------------------------")
    for m,r in results.items():
        print(f"{m} | {r['ifeval-fr']} | {r['gpqa-fr']} | {r['bac-fr']} | {r['pr-fouras']}")


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
        results[model_name]['pr-fouras'] = metrics['community:pr-fouras:0 | pr-fouras-qem'] * 100
        print_results()
    return

pipe = PipelineController(
  name="LLM Leaderboard FR Pipeline", project=project_name, version="1.0.0"
  #continue_on_fail=True,
  #continue_on_abort=True,
  #skip_children_on_fail=False,
  #skip_chlidren_on_abort=False
)

models = [
    'MaziyarPanahi/calme-3.3-qwenloi-3b',
    'MaziyarPanahi/calme-3.3-llamaloi-3b',
    'MaziyarPanahi/calme-3.3-instruct-3b',
    'MaziyarPanahi/calme-3.3-baguette-3b',
    'OpenLLM-France/Lucie-7B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct',
    'Qwen/Qwen2.5-72B-Instruct',
    'Qwen/Qwen2.5-Math-72B-Instruct',
    'AgentPublic/guillaumetell-7b',
    'allenai/Llama-3.1-Tulu-3-405B',
    'baconnier/Gaston_dolphin-2.9.1-yi-1.5-9b',
    'croissantllm/CroissantLLMChat-v0.1',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'jpacifico/Chocolatine-2-14B-Instruct-v2.0.3',
    'jpacifico/Chocolatine-2-14B-Instruct-v2.0',
    'meta-llama/Llama-3.3-70B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
    'microsoft/phi-4',
    'mistralai/Mistral-Large-Instruct-2411',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'mistralai/Mistral-Small-24B-Instruct-2501',
    'mistralai/Mixtral-8x22B-Instruct-v0.1',
    'occiglot/occiglot-7b-eu5-instruct',
    'tiiuae/Falcon3-10B-Instruct',
    'tiiuae/Falcon3-7B-Instruct',
    'utter-project/EuroLLM-9B-Instruct',
]

success = [
    'meta-llama/Llama-3.2-3B-Instruct',
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'jpacifico/Chocolatine-2-14B-Instruct-v2.0.3',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'occiglot/occiglot-7b-eu5-instruct',
    'AgentPublic/guillaumetell-7b',
    'baconnier/Gaston_dolphin-2.9.1-yi-1.5-9b',
    'MaziyarPanahi/calme-3.3-llamaloi-3b',
    'tiiuae/Falcon3-7B-Instruct',
    'tiiuae/Falcon3-10B-Instruct',
    'MaziyarPanahi/calme-3.3-qwenloi-3b',
    'OpenLLM-France/Lucie-7B-Instruct',
    'Qwen/Qwen2.5-14B-Instruct',
    'mistralai/Mistral-Small-24B-Instruct-2501',
    'MaziyarPanahi/calme-3.3-instruct-3b',
    'MaziyarPanahi/calme-3.3-baguette-3b',
    'meta-llama/Llama-3.3-70B-Instruct',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
    'Qwen/Qwen2.5-72B-Instruct',
]

oom = [
    'allenai/Llama-3.1-Tulu-3-405B',
    'mistralai/Mixtral-8x22B-Instruct-v0.1',
    'mistralai/Mistral-Large-Instruct-2411',
]

small_context = [
    'croissantllm/CroissantLLMChat-v0.1',
    'utter-project/EuroLLM-9B-Instruct',
    'Qwen/Qwen2.5-Math-72B-Instruct',
]

timeout = [
]

retry = [
]

wrong_heads = [
    'jpacifico/Chocolatine-2-14B-Instruct-v2.0',
    'microsoft/phi-4',
]

eval_tasks = [ ]

for model in success:
    task_name = f"eval_{model}"
    eval_tasks.append(task_name)
    pipe.add_step(
        name=task_name,
        parents=[ ],
        base_task_project=project_name,
        base_task_name='eval_model',
        parameter_override={
            #'General/dataset_url': '${stage_data.artifacts.dataset.url}',
            'General/model': model},
        execution_queue='abaca',
        #pre_execute_callback=pre_execute_callback_example,
        post_execute_callback=post_execute_callback,
    )

pipe.add_function_step(
    name='gather_results',
    parents=eval_tasks,
    function=gather_results,
    function_return=[],
    execution_queue='default',
)

pipe.start_locally()