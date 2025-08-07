# French LLM Evaluation Pipeline

This repository contains the evaluation pipeline for the French LLM leaderboard, supporting multiple evaluation frameworks for comprehensive model assessment.

## Overview

The pipeline evaluates Large Language Models (LLMs) on French-specific tasks and datasets, providing standardized benchmarking for the French AI community.

## Supported Frameworks

This pipeline supports two evaluation frameworks:

### 1. Lighteval (Default)
- Mature evaluation framework with vLLM integration
- Optimized for distributed evaluation and high throughput
- Full documentation in existing scripts

### 2. Inspect AI (New)
- Modern evaluation framework from UK AISI
- Focus on safety and robustness evaluation
- Clean, composable API design
- See [README_INSPECT_AI.md](README_INSPECT_AI.md) for detailed documentation

## Quick Start

### Choose Your Framework

```bash
# Use lighteval (default)
export FRAMEWORK=lighteval
./run-eval-main.sh

# Use inspect_ai
export FRAMEWORK=inspect_ai  
./run-eval-main.sh
```

### Available Tasks

Both frameworks support these French evaluation tasks:

| Task | Description |
|------|-------------|
| `ifeval-fr` | Instruction following in French |
| `gpqa-fr` | Graduate-level science questions |
| `bac-fr` | French Baccalauréat questions |
| `pr-fouras` | Père Fouras riddles |
| `sornette` | Text classification |
| `kangourou-to` | Mathematical reasoning |

## Files Structure

```
├── run-lighteval.py              # Lighteval evaluation script
├── run-inspect-ai.py             # Inspect AI evaluation script  
├── run-eval-main.sh              # Main pipeline script (dual framework)
├── tasks/
│   ├── french_evals.py           # Lighteval task definitions
│   ├── french_evals_w_reasoning.py  # Enhanced lighteval tasks
│   └── french_evals_inspect.py   # Inspect AI task definitions
├── workers_image/
│   ├── Singularityfile           # Lighteval container
│   ├── requirements.txt          # Lighteval dependencies
│   ├── Singularityfile_inspect   # Inspect AI container
│   └── requirements_inspect.txt  # Inspect AI dependencies
└── .github/workflows/
    └── build-containers.yml      # CI for container builds
```

## Usage

### Interactive Testing
```bash
# Test lighteval
python run-lighteval-interactive.py

# Test inspect_ai
python run-inspect-ai-interactive.py
```

### Production Evaluation
```bash
# SLURM cluster
export FRAMEWORK=inspect_ai
sbatch run-eval-main-slurm.sh

# Local/interactive
export FRAMEWORK=lighteval
./run-eval-main.sh
```

### ClearML Integration
Both frameworks integrate with ClearML for experiment tracking:

```python
parameters = {
    'model': 'meta-llama/Llama-3.2-3B-Instruct',
    'tasks': 'community|bac-fr|0|0,community|pr-fouras|0|0',
    'framework': 'inspect_ai',  # or 'lighteval'
    # ... other parameters
}
```

## Container Support

Two Singularity containers are automatically built via CI:
- `llm_benchmark_fr.sif` - Lighteval environment
- `llm_benchmark_fr_inspect.sif` - Inspect AI environment

The pipeline automatically selects the appropriate container based on your framework choice.

## Testing

Validate the integration:
```bash
python test_integration.py
```

## Documentation

- [README_INSPECT_AI.md](README_INSPECT_AI.md) - Detailed inspect_ai documentation
- [workers_image/](workers_image/) - Container definitions
- [tasks/](tasks/) - Task implementations for both frameworks

## Contributing

1. Add new tasks to both frameworks (`tasks/french_evals*.py` and `tasks/french_evals_inspect.py`)
2. Update container dependencies if needed
3. Test with `python test_integration.py`
4. Update documentation

## Migration

This update is fully backward compatible:
- Existing lighteval usage unchanged
- New inspect_ai framework available as opt-in
- Same datasets and evaluation targets
- Consistent ClearML integration

Choose your framework based on your evaluation needs:
- **Lighteval**: For production, high-throughput evaluation
- **Inspect AI**: For research, safety evaluation, and modern API experience