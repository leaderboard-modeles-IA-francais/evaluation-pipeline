# Inspect AI Integration

This document describes the integration of Inspect AI as an alternative evaluation framework alongside lighteval in the French LLM leaderboard evaluation pipeline.

## Overview

The pipeline now supports two evaluation frameworks:
- **lighteval**: The original framework (default)
- **inspect_ai**: The new alternative framework

Both frameworks evaluate the same French datasets but use different implementations and may provide complementary insights.

## Framework Selection

### Via Environment Variable
Set the `FRAMEWORK` environment variable before running:

```bash
# Use lighteval (default)
export FRAMEWORK=lighteval
./run-eval-main.sh

# Use inspect_ai  
export FRAMEWORK=inspect_ai
./run-eval-main.sh
```

### Via ClearML Parameters
When creating a ClearML task, set the `framework` parameter:

```python
parameters = {
    'model': 'meta-llama/Llama-3.2-3B-Instruct',
    'tasks': 'community|bac-fr|0|0,community|pr-fouras|0|0',
    'framework': 'inspect_ai',  # or 'lighteval'
    # ... other parameters
}
```

## Supported Tasks

The following French evaluation tasks are supported in both frameworks:

| Task Name | Description |
|-----------|-------------|
| `ifeval-fr` | Instruction following evaluation in French |
| `gpqa-fr` | Graduate-level questions in physics, chemistry, biology |  
| `bac-fr` | French Baccalauréat questions |
| `pr-fouras` | Père Fouras riddles |
| `sornette` | Text classification task |
| `kangourou-to` | Mathematical reasoning |

## Container Images

Two Singularity containers are provided:

- `llm_benchmark_fr.sif`: Contains lighteval and dependencies
- `llm_benchmark_fr_inspect.sif`: Contains inspect_ai and dependencies

The appropriate container is automatically selected based on the framework choice.

## Scripts

### Main Evaluation Scripts
- `run-lighteval.py`: Original lighteval implementation
- `run-inspect-ai.py`: New inspect_ai implementation  
- `run-inspect-ai-interactive.py`: Interactive version for testing

### Pipeline Scripts  
- `run-eval-main.sh`: Main pipeline script (supports both frameworks)
- `run-eval-main-slurm.sh`: SLURM version (supports both frameworks)
- `run-eval-workers.sh`: Worker node script (supports both frameworks)

## Configuration Files

### Task Definitions
- `tasks/french_evals.py`: Original lighteval tasks
- `tasks/french_evals_w_reasoning.py`: Enhanced lighteval tasks  
- `tasks/french_evals_inspect.py`: Inspect AI task implementations

### Container Definitions
- `workers_image/Singularityfile`: Lighteval container
- `workers_image/requirements.txt`: Lighteval dependencies
- `workers_image/Singularityfile_inspect`: Inspect AI container
- `workers_image/requirements_inspect.txt`: Inspect AI dependencies

## CI/CD

The GitHub Actions workflow `.github/workflows/build-containers.yml` automatically:
- Builds both container images when relevant files change
- Tests the containers to ensure they work
- Uploads artifacts for use in the cluster

## Usage Examples

### Basic Usage
```bash
# Run with lighteval (default)
python run-lighteval.py

# Run with inspect_ai
python run-inspect-ai.py

# Interactive testing with inspect_ai
python run-inspect-ai-interactive.py
```

### Full Pipeline  
```bash
# Lighteval pipeline
export FRAMEWORK=lighteval
./run-eval-main.sh

# Inspect AI pipeline  
export FRAMEWORK=inspect_ai
./run-eval-main.sh
```

### SLURM Execution
```bash
# Submit lighteval job
export FRAMEWORK=lighteval  
sbatch run-eval-main-slurm.sh

# Submit inspect_ai job
export FRAMEWORK=inspect_ai
sbatch run-eval-main-slurm.sh
```

## Differences Between Frameworks

### Lighteval
- Mature, stable framework designed for LLM evaluation
- Integrated with vLLM for efficient inference
- Supports complex parallelization and distributed evaluation  
- Rich set of built-in metrics and scorers
- Direct integration with Hugging Face ecosystem

### Inspect AI  
- Modern evaluation framework from UK AISI
- Focus on AI safety and robustness evaluation
- Clean, composable task definition API
- Built-in support for advanced evaluation patterns
- Extensible scorer and solver system

### Output Formats
Both frameworks produce results that are logged to ClearML, but with potentially different metric names and formats. The pipeline normalizes these for consistent reporting.

## Troubleshooting

### Missing Dependencies
If you see import errors:
```bash
# Install lighteval dependencies
pip install -r workers_image/requirements.txt

# Install inspect_ai dependencies  
pip install -r workers_image/requirements_inspect.txt
```

### Container Issues
If containers fail to build:
1. Check that Singularity/Apptainer is installed
2. Verify file paths in Singularityfile
3. Check that base images are accessible

### Task Loading Issues
If tasks fail to load:
1. Verify dataset paths and environment variables
2. Check that required datasets are available
3. Ensure proper authentication for private datasets

## Contributing

When adding new tasks:
1. Add the lighteval implementation to `tasks/french_evals*.py`
2. Add the inspect_ai implementation to `tasks/french_evals_inspect.py`  
3. Update both requirements files if new dependencies are needed
4. Test both implementations produce comparable results
5. Update this documentation

## Migration Guide

### From lighteval-only to dual framework
1. No changes needed for existing lighteval usage
2. Install inspect_ai to use the new framework: `pip install inspect_ai`
3. Use `framework` parameter to select evaluation framework
4. Both frameworks can be used side-by-side for comparison

### Comparing Results
To compare frameworks on the same model and tasks:
```bash
# Run both frameworks
export FRAMEWORK=lighteval && python run-lighteval.py
export FRAMEWORK=inspect_ai && python run-inspect-ai.py

# Results will be logged to ClearML with framework-specific task names
```