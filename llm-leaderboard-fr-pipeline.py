from clearml import PipelineController, Task

import pull_requests

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

eval_tasks = [ ]

models = pull_requests.models()

for model in models:
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
        execution_queue='national_clusters',
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
