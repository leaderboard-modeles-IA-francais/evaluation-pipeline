# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Custom evaluation tasks for inspect_ai.

This file implements the French evaluation tasks using the inspect_ai framework,
equivalent to the lighteval tasks defined in french_evals.py.

See: https://huggingface.co/fr-gouv-coordination-ia
"""

import os
import random
import re
import string
from pathlib import Path
from typing import Any

try:
    from inspect_ai import Task, eval, task
    from inspect_ai.dataset import Dataset, Sample
    from inspect_ai.model import ChatMessageUser, GenerateConfig
    from inspect_ai.scorer import (
        Scorer, 
        Score, 
        Target, 
        accuracy, 
        scorer, 
        mean
    )
    from inspect_ai.solver import (
        chain_of_thought,
        generate, 
        multiple_choice,
        Solver
    )
    INSPECT_AI_AVAILABLE = True
except ImportError:
    print("Warning: inspect_ai not available. Install with: pip install inspect_ai")
    INSPECT_AI_AVAILABLE = False
    # Define dummy classes for compatibility
    class Task: pass
    class Dataset: pass
    class Sample: pass
    def task(fn): return fn
    def scorer(*args, **kwargs): return lambda fn: fn

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: datasets not available. Install with: pip install datasets")
    DATASETS_AVAILABLE = False
    def load_dataset(*args, **kwargs): 
        raise ImportError("datasets package not installed")


# Normalization functions (ported from lighteval version)
def helm_normalizer_fr(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Adapted for pr_fouras dataset"""

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
    """Custom normalizer for bac-fr dataset"""

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


# Custom scorers
if INSPECT_AI_AVAILABLE:
    @scorer(metrics=[mean()])
    def bac_prefix_suffix_match() -> Scorer:
        """Custom scorer for BAC-fr that allows prefix, suffix, or exact match"""
        
        def score(state, target: Target):
            if not state.output.completion:
                return Score(value=0.0)
            
            pred = math_normalizer(state.output.completion.strip())
            gold = math_normalizer(target.text.strip())
            
            if pred.startswith(gold) or pred.endswith(gold) or gold == pred:
                return Score(value=1.0)
            
            # Last chance with split on '='
            gold_lc = gold.split('=')[-1]
            pred_lc = pred.split('=')[-1]
            if pred_lc.startswith(gold_lc) or pred_lc.endswith(gold_lc) or gold_lc == pred_lc:
                return Score(value=1.0)
            
            return Score(value=0.0)
        
        return score


    @scorer(metrics=[mean()])
    def pr_fouras_prefix_suffix_match() -> Scorer:
        """Custom scorer for pr-fouras that handles multiple responses separated by /"""
        
        def score(state, target: Target):
            if not state.output.completion:
                return Score(value=0.0)
            
            # Split multi-responses
            predictions = state.output.completion.split('/')
            
            gold = helm_normalizer_fr(target.text.strip())
            
            for pred in predictions:
                pred = helm_normalizer_fr(pred.strip())
                if pred.startswith(gold) or pred.endswith(gold) or gold == pred:
                    return Score(value=1.0)
            
            return Score(value=0.0)
        
        return score
else:
    def bac_prefix_suffix_match(): 
        return None
    def pr_fouras_prefix_suffix_match(): 
        return None


# Dataset loading functions
def load_hf_dataset(repo_name: str, subset: str = "default") -> Dataset:
    """Load a Hugging Face dataset and convert to inspect_ai format"""
    dsdir = Path(os.getenv("DATASETS_DIRECTORY", "fr-gouv-coordination-ia"))
    hf_dataset = load_dataset(str(dsdir / repo_name), subset, split="train")
    
    samples = []
    for item in hf_dataset:
        # Convert to inspect_ai Sample format
        # This will be customized per task type
        sample = Sample(
            input=str(item),  # Placeholder - will be overridden by task-specific functions
            target=""  # Placeholder - will be overridden by task-specific functions
        )
        samples.append(sample)
    
    return Dataset(samples)


# Task definitions
@task
def ifeval_fr():
    """IFEval-fr task - Instruction following evaluation in French"""
    
    def create_samples():
        dsdir = Path(os.getenv("DATASETS_DIRECTORY", "fr-gouv-coordination-ia"))
        hf_dataset = load_dataset(str(dsdir / "IFEval-fr"), "default", split="train")
        
        samples = []
        for item in hf_dataset:
            sample = Sample(
                input=item["prompt"],
                target="",  # IFEval uses custom evaluation
                metadata={
                    "instructions_id_list": item["instruction_id_list"],
                    "kwargs": item["kwargs"]
                }
            )
            samples.append(sample)
        
        return Dataset(samples)
    
    return Task(
        dataset=create_samples(),
        solver=generate(),
        scorer=accuracy(),  # TODO: Implement IFEval-specific scorer
        config=GenerateConfig(max_tokens=2048)
    )


@task
def gpqa_fr():
    """GPQA-fr task - Graduate-level physics, chemistry, and biology questions in French"""
    
    def create_samples():
        dsdir = Path(os.getenv("DATASETS_DIRECTORY", "fr-gouv-coordination-ia"))
        hf_dataset = load_dataset(str(dsdir / "gpqa-fr"), "default", split="train")
        
        samples = []
        for item in hf_dataset:
            # Randomize answer choices
            choices = [
                item["Réponse incorrecte 1"], 
                item["Réponse incorrecte 2"], 
                item["Réponse incorrecte 3"]
            ]
            correct_idx = random.randint(0, 3)
            choices.insert(correct_idx, item["Réponse correcte"])
            
            prompt = f"Question: {item['Question']}\n\n"
            prompt += "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            prompt += "\nRéponse: "
            
            sample = Sample(
                input=prompt,
                target=chr(65 + correct_idx),
                choices=choices
            )
            samples.append(sample)
        
        return Dataset(samples)
    
    return Task(
        dataset=create_samples(),
        solver=multiple_choice(),
        scorer=accuracy(),
        config=GenerateConfig(max_tokens=1)
    )


@task
def bac_fr():
    """BAC-fr task - French Baccalauréat questions"""
    
    def create_samples():
        dsdir = Path(os.getenv("DATASETS_DIRECTORY", "fr-gouv-coordination-ia"))
        hf_dataset = load_dataset(str(dsdir / "bac-fr"), "default", split="train")
        
        samples = []
        for item in hf_dataset:
            prompt = "Répondre exactement à la question en suivant les instructions.\n\n"
            if item.get('instruction'):
                prompt += f"Instruction: {item['instruction']}\n\n"
            prompt += f"Question: {item['enonce']}\n"
            prompt += "Réponse: "
            
            if item.get("choix"):  # Multiple choice
                choices = item["choix"] if isinstance(item["choix"], list) else [item["choix"]]
                correct_idx = choices.index(item["choix correct"])
                
                sample = Sample(
                    input=prompt,
                    target=chr(65 + correct_idx),
                    choices=choices
                )
            else:  # Open-ended
                sample = Sample(
                    input=prompt,
                    target=item["reponse"]
                )
            
            samples.append(sample)
        
        return Dataset(samples)
    
    return Task(
        dataset=create_samples(),
        solver=generate(),
        scorer=bac_prefix_suffix_match(),
        config=GenerateConfig(max_tokens=2048)
    )


@task
def pr_fouras():
    """Père Fouras riddles task"""
    
    def create_samples():
        dsdir = Path(os.getenv("DATASETS_DIRECTORY", "fr-gouv-coordination-ia"))
        hf_dataset = load_dataset(str(dsdir / "pr-fouras"), "default", split="train")
        
        samples = []
        for item in hf_dataset:
            prompt = "Trouver la réponse exacte à l'énigme. Vous pouvez proposer plusieurs réponses possibles. "
            prompt += "Chaque réponse doit être séparée d'un caractère /.\n"
            prompt += "Exemple:\nEnigme: Plus je travaille, plus je raccourcis. Qui suis-je ?\n"
            prompt += "Réponses: Des ciseaux / Une paire de ciseaux / Une bougie / Une gomme / "
            prompt += "Une personne agée / Un vieux / Un vêtement / Un sécateur / Un clou / Une pause.\n\n"
            prompt += f"Enigme: {item['enigme']}\n"
            prompt += "Réponses: "
            
            sample = Sample(
                input=prompt,
                target=item["reponse"]
            )
            samples.append(sample)
        
        return Dataset(samples)
    
    return Task(
        dataset=create_samples(),
        solver=generate(),
        scorer=pr_fouras_prefix_suffix_match(),
        config=GenerateConfig(max_tokens=2048)
    )


@task 
def sornette():
    """Sornette task - Text classification"""
    
    def create_samples():
        dsdir = Path(os.getenv("DATASETS_DIRECTORY", "fr-gouv-coordination-ia"))
        hf_dataset = load_dataset(str(dsdir / "sornette"), "default", split="train")
        
        samples = []
        for item in hf_dataset:
            choices = ['burlesque et fantaisiste', 'ludique et didactique', 'insidieux et mensonger', 'moral et accablant']
            random.shuffle(choices)
            correct_idx = choices.index(item["gold"])
            
            prompt = f"Texte: {item['text']}\n\n"
            prompt += "Question: Le texte est-il:\n"
            prompt += "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            prompt += "\nRéponse: "
            
            sample = Sample(
                input=prompt,
                target=chr(65 + correct_idx),
                choices=choices
            )
            samples.append(sample)
        
        return Dataset(samples)
    
    return Task(
        dataset=create_samples(),
        solver=multiple_choice(),
        scorer=accuracy(),
        config=GenerateConfig(max_tokens=1)
    )


@task
def kangourou_to():
    """Kangourou-to task - Mathematical reasoning"""
    
    def create_samples():
        dsdir = Path(os.getenv("DATASETS_DIRECTORY", "fr-gouv-coordination-ia"))
        hf_dataset = load_dataset(str(dsdir / "kangourou-to"), "default", split="train")
        
        samples = []
        for item in hf_dataset:
            choices = item["choices"].copy()
            random.shuffle(choices)
            correct_idx = choices.index(item["gold"])
            
            prompt = f"Question: {item['question']}\n\n"
            prompt += "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            prompt += "\nRéponse: "
            
            sample = Sample(
                input=prompt,
                target=chr(65 + correct_idx),
                choices=choices
            )
            samples.append(sample)
        
        return Dataset(samples)
    
    return Task(
        dataset=create_samples(),
        solver=multiple_choice(),
        scorer=accuracy(),
        config=GenerateConfig(max_tokens=1)
    )


# Task registry
AVAILABLE_TASKS = {
    "ifeval-fr": ifeval_fr,
    "gpqa-fr": gpqa_fr, 
    "bac-fr": bac_fr,
    "pr-fouras": pr_fouras,
    "sornette": sornette,
    "kangourou-to": kangourou_to
}