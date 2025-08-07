"""
French evaluation tasks for inspect_ai

This module defines inspect_ai implementations of French language benchmarks
for LLM evaluation, providing an alternative to the lighteval framework.

Tasks implemented using patterns from clebreto/inspect_evals fork:
- ifeval_fr: French instruction following evaluation  
- gpqa_fr: French graduate-level science questions
- bac_fr: French Baccalauréat questions (with custom math scorer)
- pr_fouras: Père Fouras riddles (with multi-response scorer)
- sornette: Text classification
- kangourou_to: Mathematical reasoning

Usage:
    inspect eval tasks.french_evals_inspect:ifeval_fr
    inspect eval tasks.french_evals_inspect:gpqa_fr
"""

import re
import string
from typing import Any, cast

import numpy as np
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    choice,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState, generate, multiple_choice


# French-specific normalization functions

def helm_normalizer_fr(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Adapted for French text processing."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(le |la |les |l |un |une |des )", "", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def homogeneize_numbers(text: str) -> str:
        try:
            return str(float(text.replace(',', '.')))
        except ValueError:
            return text

    def remove_punc(text: str) -> str:
        punctuation = string.punctuation
        regex = rf'(?<!\d)[{punctuation}](?!\d)|(?<=\D)[{punctuation}]+|[{punctuation}]+(?=\D)|(?<=^)[{punctuation}]+|[{punctuation}]+(?=$)'
        return re.sub(regex, "", text)

    def remove_plural_feminine(text: str) -> str:
        regex = r'\b(\w+?)(?:es|e|s|x)\b'
        return re.sub(regex, r'\1', text)

    def lower(text: str) -> str:
        return text.lower()

    def _tokenize(text):
        text = lower(text)
        text = re.sub(r"['\''`]", " ", text)
        text = remove_articles(text)
        return re.split(" ", text)

    tokens = [remove_plural_feminine(remove_punc(white_space_fix(homogeneize_numbers(t)))) for t in _tokenize(text)]
    return " ".join([t for t in tokens if t != ""]).strip()


def math_normalizer(text: str) -> str:
    """Custom normalizer for mathematical text (bac-fr, kangourou-to)"""

    def space_digit(text: str) -> str:
        return re.sub(r"(\d+[.]*\d*)", r" \1 ", text)

    def homogeneize_numbers(text: str) -> str:
        try:
            return str(round(float(text.replace(',', '.')), 2))
        except ValueError:
            return text

    def lower(text: str) -> str:
        return text.lower()

    def remove_punc(text: str) -> str:
        punctuation = "'`*\"~.()$"
        regex = rf'(?<=^)[{punctuation}]+|[{punctuation}]+(?=$)'
        return re.sub(regex, "", text)

    def remove_special_chars(text: str) -> str:
        regex = r"[\"'`]|(?:\n)|(?:\t)|(?:\r)|(?:\\n)|(?:\\t)|(?:\\r)"
        return re.sub(regex, "", text)

    def comma_to_point(text: str) -> str:
        return re.sub(r"(\d+),(\d+)", r"\1.\2", text)

    def clean_math_expressions(text: str) -> str:
        text = re.sub(r"[{}]", "", text)
        regex = r"(\\*)(boxed|fbox|underline)"
        text = re.sub(regex, "", text)
        return text

    def _tokenize(text):
        text = lower(text)
        text = comma_to_point(text)
        text = clean_math_expressions(text)
        text = remove_punc(text)
        text = remove_special_chars(space_digit(text))
        return text.split()

    tokens = [homogeneize_numbers(t) for t in _tokenize(text)]
    return "".join([t for t in tokens if t != ""]).strip()


# Custom metrics and scorers

@metric
def french_accuracy() -> Metric:
    """French-aware accuracy metric"""
    def metric(scores: list[SampleScore]) -> Value:
        correct = sum(1 for score in scores if score.score.value == 1)
        return correct / len(scores) if scores else 0.0
    return metric


@scorer(metrics=[french_accuracy()])
def math_prefix_suffix_match() -> Scorer:
    """Custom scorer for mathematical tasks that allows prefix, suffix, or exact match"""
    
    async def score(state: TaskState, target: Target) -> Score:
        if not state.output.completion:
            return Score(value=0, answer="")
        
        pred = math_normalizer(state.output.completion.strip())
        gold = math_normalizer(target.text.strip())
        
        if pred.startswith(gold) or pred.endswith(gold) or gold == pred:
            return Score(value=1, answer=state.output.completion)
        
        # Last chance with split on '='
        gold_lc = gold.split('=')[-1] if '=' in gold else gold
        pred_lc = pred.split('=')[-1] if '=' in pred else pred
        if pred_lc.startswith(gold_lc) or pred_lc.endswith(gold_lc) or gold_lc == pred_lc:
            return Score(value=1, answer=state.output.completion)
        
        return Score(value=0, answer=state.output.completion)
    
    return score


@scorer(metrics=[french_accuracy()])
def multi_response_match() -> Scorer:
    """Custom scorer for pr-fouras that handles multiple responses separated by /"""
    
    async def score(state: TaskState, target: Target) -> Score:
        if not state.output.completion:
            return Score(value=0, answer="")
        
        # Split multi-responses
        predictions = state.output.completion.split('/')
        
        gold = helm_normalizer_fr(target.text.strip())
        
        for pred in predictions:
            pred = helm_normalizer_fr(pred.strip())
            if pred.startswith(gold) or pred.endswith(gold) or gold == pred:
                return Score(value=1, answer=state.output.completion)
        
        return Score(value=0, answer=state.output.completion)
    
    return score


@scorer(metrics=[french_accuracy()])
def french_exact_match() -> Scorer:
    """French-aware exact match scorer"""
    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion.strip()
        expected = target.text
        
        # Normalize both texts using helm_normalizer_fr
        norm_answer = helm_normalizer_fr(answer)
        norm_expected = helm_normalizer_fr(expected)
        
        return Score(
            value=1 if norm_answer == norm_expected else 0,
            answer=answer
        )
    return score


# IFEval-specific implementation (from clebreto fork)

@metric
def if_metric() -> Metric:
    def _final_accuracy_stderr(
        scores: list[SampleScore], mean_final_accuracy: float
    ) -> float:
        total_num_instructions = int(
            sum(
                cast(dict[str, Any], score.score.value)["num_instructions"]
                for score in scores
            )
        )
        mean_num_instructions = total_num_instructions / len(scores)
        variance = 0.0
        cluster_count = len(scores)
        for score in scores:
            value = cast(dict[str, Any], score.score.value)
            inst_level_strict = int(value["inst_level_strict"])
            inst_level_loose = int(value["inst_level_loose"])
            prompt_level_strict = int(value["prompt_level_strict"])
            prompt_level_loose = int(value["prompt_level_loose"])
            num_instructions = int(value["num_instructions"])

            loose_only = inst_level_loose - inst_level_strict
            num_incorrect = int(num_instructions - inst_level_loose)
            prompt_adjustment = (
                0.25
                * (prompt_level_strict + prompt_level_loose)
                * mean_num_instructions
                / num_instructions
            )
            vector = [
                (0.5 + prompt_adjustment - mean_final_accuracy) * inst_level_strict,
                (0.25 + prompt_adjustment - mean_final_accuracy) * loose_only,
                (0.0 + prompt_adjustment - mean_final_accuracy) * num_incorrect,
            ]
            variance += np.outer(vector, vector).sum()

        stderr = (
            np.sqrt(variance * cluster_count / (cluster_count - 1))
            / total_num_instructions
            if cluster_count > 1
            else 0.0
        )

        return stderr

    def metric(scores: list[SampleScore]) -> Value:
        statistics: list[float] = []
        prompt_keys = ["prompt_level_strict", "prompt_level_loose"]
        instruct_keys = ["inst_level_strict", "inst_level_loose"]
        final_keys = [
            "prompt_strict_acc",
            "prompt_strict_stderr",
            "prompt_loose_acc",
            "prompt_loose_stderr",
            "inst_strict_acc",
            "inst_strict_stderr",
            "inst_loose_acc",
            "inst_loose_stderr",
            "final_acc",
            "final_stderr",
        ]

        # calculate prompt-level accuracies + stderrs
        for key in prompt_keys:
            score_lst = [
                cast(dict[str, Any], score.score.value)[key] for score in scores
            ]
            statistics.append(np.mean(score_lst).item())
            stderr = (
                np.std(score_lst, ddof=1).item() / np.sqrt(len(score_lst))
                if len(score_lst) > 1
                else 0.0
            )
            statistics.append(stderr)

        # calculate instruction-level accuracies + clustered stderrs
        for key in instruct_keys:
            flattened = []
            for score in scores:
                value = cast(dict[str, Any], score.score.value)
                num_correct = int(value[key])
                num_incorrect = int(value["num_instructions"] - value[key])
                flattened.extend([True] * num_correct + [False] * num_incorrect)

            mean = np.mean(flattened).item()
            statistics.append(mean)

            # Because the inclusion of instructions are correlated, we need to cluster
            # the standard errors by prompt. The clustered calculation follows the logic
            # in the main stderr(cluster="cluster") code found in inspect_ai.scorer.
            variance = 0.0
            cluster_count = len(scores)
            for score in scores:
                value = cast(dict[str, Any], score.score.value)
                num_correct = int(value[key])
                num_incorrect = int(value["num_instructions"] - value[key])
                vector = [num_correct * (1 - mean), num_incorrect * (0 - mean)]
                variance += np.outer(vector, vector).sum()

            stderr = (
                np.sqrt(variance * cluster_count / (cluster_count - 1)) / len(flattened)
                if cluster_count > 1
                else 0.0
            )
            statistics.append(stderr)

        # Calculate the final accuracy and its standard error
        statistics.append(
            np.mean([statistics[i] for i in range(0, len(statistics), 2)]).item()
        )
        statistics.append(_final_accuracy_stderr(scores, statistics[-1]))

        return {k: v for k, v in zip(final_keys, statistics, strict=True)}

    return metric


@scorer(metrics=[if_metric()])
def instruction_following() -> Scorer:
    from instruction_following_eval.evaluation import (  # type: ignore
        InputExample,
        ensure_nltk_resource,
        test_instruction_following,
    )

    ensure_nltk_resource()  # Required before calling test_instruction_following

    async def score(state: TaskState, target: Target) -> Score:
        # construct the input to IFEval's evaluation functions using the data class
        eval_input = InputExample(
            key=state.sample_id,
            instruction_id_list=state.metadata["instruction_id_list"],
            prompt=state.metadata["prompt"],
            kwargs=state.metadata["kwargs"],
        )

        # retrieve evaluated outputs
        out_strict = test_instruction_following(
            eval_input, state.output.completion, strict=True
        )
        out_loose = test_instruction_following(
            eval_input, state.output.completion, strict=False
        )
        ret_value = {
            "prompt_level_strict": out_strict.follow_all_instructions,
            "inst_level_strict": sum(out_strict.follow_instruction_list),
            "prompt_level_loose": out_loose.follow_all_instructions,
            "inst_level_loose": sum(out_loose.follow_instruction_list),
            "num_instructions": len(out_loose.follow_instruction_list),
        }

        # return score with resulting outputs, model answer, and the
        # expected instructions
        return Score(
            value=ret_value,
            answer=state.output.completion,
            explanation=" ".join(state.metadata["instruction_id_list"]),
        )

    return score


# Task implementations

@task
def ifeval_fr() -> Task:
    """French instruction following evaluation"""
    return Task(
        dataset=hf_dataset(
            path="fr-gouv-coordination-ia/IFEval-fr",
            split="train",
            sample_fields=ifeval_record_to_sample
        ),
        solver=[generate()],
        scorer=instruction_following(),
    )


def ifeval_record_to_sample(record: dict[str, Any]) -> Sample:
    new_kwargs = {}
    for index in range(len(record["instruction_id_list"])):
        # remove None values from kwargs to avoid unexpected keyword argument errors
        # in build_description method from the IFEval package.
        kwargs = {k: v for k, v in record["kwargs"][index].items() if v}
        new_kwargs[index] = kwargs

    return Sample(
        id=record["key"],
        input=record["prompt"],
        metadata={
            "prompt": record["prompt"],
            "instruction_id_list": record["instruction_id_list"],
            "kwargs": new_kwargs,
        },
    )


@task  
def gpqa_fr() -> Task:
    """French graduate-level science questions"""
    return Task(
        dataset=hf_dataset(
            path="fr-gouv-coordination-ia/gpqa-fr",
            split="train", 
            sample_fields=gpqa_record_to_sample
        ),
        solver=[multiple_choice(shuffle=True)],
        scorer=choice(),
        config=GenerateConfig(temperature=0.5),
    )


def gpqa_record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["Question"],
        choices=[
            str(record["Réponse correcte"]),
            str(record["Réponse incorrecte 1"]), 
            str(record["Réponse incorrecte 2"]),
            str(record["Réponse incorrecte 3"]),
        ],
        target="A",  # Correct answer is always first, shuffling handled by solver
    )


@task
def bac_fr() -> Task:
    """French Baccalauréat questions"""
    return Task(
        dataset=hf_dataset(
            path="fr-gouv-coordination-ia/bac-fr",
            split="test",
            sample_fields=bac_record_to_sample
        ),
        solver=[generate()],
        scorer=math_prefix_suffix_match(),
    )


def bac_record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["question"],
        target=str(record["answer"]),
    )


@task
def pr_fouras() -> Task:
    """Père Fouras riddles"""
    return Task(
        dataset=hf_dataset(
            path="fr-gouv-coordination-ia/pr-fouras",
            split="test", 
            sample_fields=fouras_record_to_sample
        ),
        solver=[generate()],
        scorer=multi_response_match(),
    )


def fouras_record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["question"],
        target=record["answer"],
    )


@task
def sornette() -> Task:
    """Text classification task"""
    return Task(
        dataset=hf_dataset(
            path="fr-gouv-coordination-ia/sornette",
            split="test",
            sample_fields=sornette_record_to_sample
        ),
        solver=[generate()],
        scorer=french_exact_match(),
    )


def sornette_record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["text"],
        target=str(record["label"]),
    )


@task
def kangourou_to() -> Task:
    """Mathematical reasoning task"""
    return Task(
        dataset=hf_dataset(
            path="fr-gouv-coordination-ia/kangourou-to", 
            split="test",
            sample_fields=kangourou_record_to_sample
        ),
        solver=[generate()],
        scorer=math_prefix_suffix_match(),
    )


def kangourou_record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["question"],
        target=str(record["answer"]),
    )


# Task registry for backwards compatibility
AVAILABLE_TASKS = {
    "ifeval-fr": ifeval_fr,
    "gpqa-fr": gpqa_fr, 
    "bac-fr": bac_fr,
    "pr-fouras": pr_fouras,
    "sornette": sornette,
    "kangourou-to": kangourou_to
}