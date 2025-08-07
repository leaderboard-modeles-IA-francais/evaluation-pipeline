# French Language Model Evaluation Pipeline

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Overview

This is a French language model evaluation pipeline using ClearML for orchestration, lighteval for evaluation, VLLM for model inference, and Ray for distributed computing. It evaluates French language models on custom benchmark tasks and publishes results to HuggingFace Hub repositories.

## Working Effectively

### Prerequisites and Environment Setup

- **NEVER CANCEL builds or long-running operations** - evaluation jobs can take 2+ hours
- Set timeout to 180+ minutes (10800+ seconds) for full evaluation runs
- Python 3.12+ is required for all components
- Container runtime needed: Docker (preferred) or Apptainer/Singularity
- Git and basic system tools are available

### Essential Dependencies Installation

**CRITICAL: Installation takes 45-90 minutes. NEVER CANCEL. Set timeout to 120+ minutes.**

```bash
# Create and use a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install core dependencies - takes 45-90 minutes, NEVER CANCEL
pip install --upgrade pip
pip install -r workers_image/requirements.txt
```

**Note**: If pip installation fails due to network timeouts, this is expected in restricted environments. The pipeline is designed to run in containerized environments on HPC clusters.

### Container-based Setup (Recommended)

Build the evaluation environment container:

```bash
# Build container - takes 15-30 minutes, NEVER CANCEL
docker build -f workers_image/Singularityfile -t llm-eval-fr .

# Or use Apptainer/Singularity if available
apptainer build llm_benchmark_fr.sif workers_image/Singularityfile
```

### Required Environment Variables

Set these environment variables before running evaluations:

```bash
# HuggingFace authentication (required)
export HF_TOKEN="your_hf_token_here"
echo $HF_TOKEN > ~/.hf_token

# HuggingFace Hub publishing (optional)
export HF_USER_ACCESS_GIT="your_hf_username"  
export HF_TOKEN_ACCESS_GIT="your_hf_git_token"
echo $HF_USER_ACCESS_GIT > ~/.hf_push_user
echo $HF_TOKEN_ACCESS_GIT > ~/.hf_push_token

# ClearML configuration (for pipeline orchestration)
export CLEARML_WEB_HOST="your_clearml_host"
export CLEARML_API_HOST="your_clearml_api_host"
export CLEARML_FILES_HOST="your_clearml_files_host"
export CLEARML_API_ACCESS_KEY="your_access_key"
export CLEARML_API_SECRET_KEY="your_secret_key"

# Temporary directories for large files
export TMP_DIR="/tmp/${USER}-runtime-dir"
mkdir -p ${TMP_DIR}
export HF_HOME="${TMP_DIR}/cache"
export TRITON_CACHE_DIR="${TMP_DIR}/triton"  
export VLLM_CONFIG_ROOT="${TMP_DIR}/.config/vllm"
```

## Running Evaluations

### Interactive Evaluation (Single Model)

**NEVER CANCEL: Takes 60-180 minutes per model. Set timeout to 240+ minutes.**

```bash
# Activate environment
source venv/bin/activate

# Run interactive evaluation
python3 run-lighteval-interactive.py

# Or for non-reasoning models
# Edit tasks/french_evals.py to customize tasks and models
python3 run-lighteval.py
```

### Pipeline Evaluation (Multiple Models)

**NEVER CANCEL: Takes 120-480 minutes for full pipeline. Set timeout to 600+ minutes.**

```bash
# Run ClearML-orchestrated pipeline
python3 llm-leaderboard-fr-pipeline.py

# For specific clusters (musa, chuc, auto)
python3 llm-leaderboard-fr-pipeline-musa.py
python3 llm-leaderboard-fr-pipeline-chuc.py
```

### HPC Cluster Execution

For distributed execution on HPC clusters with Slurm/OAR:

```bash
# Submit to cluster - execution time varies by queue (30 minutes to 8+ hours)
# NEVER CANCEL - cluster jobs have their own timeouts
./run-eval-main-slurm.sh

# Interactive cluster run
./run-eval-main-interactive.sh

# Direct execution (requires cluster nodes)
./run-eval-main.sh
```

### Container-based Execution

```bash
# Using Docker - takes 60-180 minutes, NEVER CANCEL
docker run --gpus all -v $(pwd):/workspace -v ~/.hf_token:/root/.hf_token \
  llm-eval-fr python3 run-lighteval.py

# Using Apptainer/Singularity (HPC environments)  
apptainer exec --nv llm_benchmark_fr.sif python3 run-lighteval.py
```

## Validation

### Always Test Before Committing

```bash
# Validate all Python scripts (takes 30 seconds)
python3 -m py_compile run-lighteval.py
python3 -m py_compile run-lighteval-interactive.py  
python3 -m py_compile llm-leaderboard-fr-pipeline.py
python3 -m py_compile tasks/french_evals.py
python3 -m py_compile tasks/french_evals_w_reasoning.py

# Validate shell scripts (takes 5 seconds)
bash -n run-eval-main.sh
bash -n run-eval-main-interactive.sh
bash -n run-eval-main-slurm.sh
bash -n run-eval-workers.sh

# Test Docker build (takes 15-30 minutes, NEVER CANCEL)
docker build -t eval-test .
```

### Manual Validation Scenarios

**ALWAYS test actual functionality after making changes:**

1. **Basic Script Test**: Verify Python syntax and imports work
2. **Container Build Test**: Ensure Docker/Apptainer builds complete successfully  
3. **Evaluation Task Test**: Run evaluation on a small model subset (30-60 minutes)
4. **Result Publishing Test**: Verify results can be pushed to HuggingFace Hub

### Expected Results

- Evaluation results saved to `/tmp/${USER}-runtime-dir/results/`
- Detailed results in parquet format in `details/` subdirectory
- Summary results in JSON format in main results directory
- Results automatically pushed to HuggingFace Hub if configured

## Common Tasks

### Modify Evaluation Tasks

Edit task definitions in:
- `tasks/french_evals.py` - Standard evaluation tasks
- `tasks/french_evals_w_reasoning.py` - Reasoning-focused tasks

Key French benchmarks included:
- `bac-fr` - French BAC exam questions
- `ifeval-fr` - French instruction following
- `pr-fouras` - French reasoning tasks  
- `gpqa-fr` - French scientific Q&A

### Add New Models

Edit model lists in:
- `run-lighteval-interactive.py` (lines 84-94)
- `pull_requests.py` - handles model discovery from PR requests
- ClearML pipeline scripts retrieve models automatically

### Debug Evaluations

Common debugging steps:

```bash
# Check Ray cluster status (if using distributed execution)
ray status

# Monitor GPU usage
nvidia-smi

# Check evaluation logs
tail -f /tmp/.clearml_agent_out.*

# Validate HuggingFace token access
huggingface-cli whoami

# Test model loading
python3 -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct'))"
```

### Result Management

```bash
# Push results manually
python3 push_results.py /path/to/results repo-name

# Pull pending evaluation requests  
python3 pull_requests.py

# Check result structure
ls -la /tmp/${USER}-runtime-dir/results/
```

## Repository Structure

```
├── llm-leaderboard-fr-pipeline*.py    # ClearML pipeline definitions
├── run-eval-main*.sh                  # HPC cluster execution scripts  
├── run-eval-workers.sh                # Distributed worker setup
├── run-lighteval*.py                  # Core evaluation scripts
├── tasks/
│   ├── french_evals.py                # Standard French evaluation tasks
│   └── french_evals_w_reasoning.py    # Reasoning evaluation tasks
├── workers_image/
│   ├── Singularityfile                # Container definition
│   ├── requirements.txt               # Python dependencies
│   └── setup_nltk.py                  # NLTK data setup
├── push_results.py                    # Result publishing utility
└── pull_requests.py                   # Model request management
```

## Critical Timing Information

**NEVER CANCEL any of these operations:**

- **Dependency Installation**: 45-90 minutes (set timeout: 120+ minutes)
- **Container Build**: 15-30 minutes (set timeout: 45+ minutes)
- **Single Model Evaluation**: 60-180 minutes (set timeout: 240+ minutes)
- **Full Pipeline**: 120-480 minutes (set timeout: 600+ minutes)
- **HPC Cluster Jobs**: 30 minutes to 8+ hours (managed by cluster scheduler)

## Troubleshooting

- **Pip install timeouts**: Expected in restricted environments - use containers
- **CUDA out of memory**: Reduce `gpu_memory_utilization` parameter (default 0.5)
- **Ray connection issues**: Check firewall settings, ensure proper port access  
- **HuggingFace token errors**: Verify token has model access permissions
- **Container build fails**: Check available disk space (needs 10+ GB)

## Development Workflow

1. **Always validate syntax** before running evaluations
2. **Test with small model subsets** before full pipeline runs
3. **Monitor resource usage** during evaluations  
4. **Backup results** before making changes
5. **Use version control** for task modifications

Remember: This pipeline is designed for long-running, resource-intensive evaluations. Plan accordingly and never cancel operations prematurely.