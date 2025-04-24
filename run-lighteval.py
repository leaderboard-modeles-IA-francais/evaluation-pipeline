import os
from pathlib import Path
from clearml import Task
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig

def main():
    unix_user=os.environ.get("USER")
    output_dir=f"/tmp/{unix_user}-runtime-dir/results"
    #tasks_path=Path(f"tasks/french_evals.py")
    tasks_path=Path(f"tasks/french_evals_w_reasoning.py")

    # FIXME get task by id instead
    task = Task.init(project_name = "LLM Leaderboard FR", task_name = "eval_model")

    # FIXME
    # vllm_model.py:211: FutureWarning: It is strongly recommended to run mistral models with `--tokenizer-mode "mistral"` to ensure correct encoding and decoding.

    parameters = {
        'model': 'meta-llama/Llama-3.2-3B-Instruct',
        'dtype': 'bfloat16',
        'gpu_memory_utilization': 0.5,
        'nb_gpus_per_node': 4,
        'nb_nodes': 1,
        'enforce_eager': True,
        'tasks': 'community|bac-fr|0|0,community|ifeval-fr|0|0,community|pr-fouras|0|0,community|gpqa-fr|0|0',
        'max_model_length': 8192,
        'use_chat_template': True,
    }

    task.connect(parameters)

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=True,
        push_to_hub=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory=tasks_path,
        use_chat_template=parameters['use_chat_template'],
        #env_config=EnvConfig(cache_dir="tmp/"),
        # Remove the 2 parameters below once your configuration is tested
        #override_batch_size=1,
        #max_samples=10
)

    model_config = VLLMModelConfig(
        pretrained=parameters['model'],
        dtype=parameters['dtype'],
        gpu_memory_utilization=parameters['gpu_memory_utilization'],
        tensor_parallel_size=parameters['nb_nodes'] * parameters['nb_gpus_per_node'],
        enforce_eager=parameters['enforce_eager'],
        max_model_length=parameters['max_model_length'],
        use_chat_template=parameters['use_chat_template'],
    )

    pipeline = Pipeline(
        tasks=parameters["tasks"],
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

    final = pipeline.get_results()
    logger = task.get_logger()
    print(final)
    for task_name, metrics in final['results'].items():
        for k, v in metrics.items():
            logger.report_single_value(f"{task_name} | {k}", v)

            task.upload_artifact(name='results', artifact_object=output_dir)

# Run the main function
if __name__ == "__main__":
    main()
