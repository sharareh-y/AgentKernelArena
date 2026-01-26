# AgentKernelArena: AI Agent Evaluation for GPU Kernel Optimization

AgentKernelArena is an AI agent evaluation framework and harness that makes it easy for you to compare LLMs and agents -- such as Cursor, Claude Code, or OpenEvolve (GEAK) -- on important GPU kernel optimization tasks. Automatically benchmark you agents against GPU programming challenges. Get metrics that are critical to measuring the goodness of generated kernels: compilation success (it runs), correctness (it runs correctly), and GPU performance (it runs faster than the baseline). Built by AMD's AI Group (AIG).

## Overview

AgentKernelArena enables systematic evaluation of AI agents on GPU kernel optimization tasks by:
- Supporting multiple agent architectures (cursor, claude_code, swe-agent, single_llm_call)
- Providing isolated workspaces for reproducible testing
- Automating compilation, correctness testing, and performance profiling
- Generating comprehensive evaluation reports with quantitative scoring

## Features

- **Multi-Agent Support**: Cursor, Claude Code, SWE-agent, OpenEvolve (GEAK), single LLM calls, and custom agents
- **Multiple LLM Providers**: Integration with OpenAI (GPT-5), Anthropic Claude (Opus 4, Sonnet 4.5), OpenRouter, and vLLM
- **Task Categories**: Support for HIP, Triton (TritonBench, ROCmBench), and PyTorch kernel optimization tasks
- **Automated Scoring**: Cumulative scoring based on compilation (20 points), correctness (100 points), and speedup (ratio × 100 points)
- **Workspace Isolation**: Each task runs in a timestamped duplicate workspace for reproducibility
- **Comprehensive Logging**: Detailed logs with timestamps, prompts, and results for every task execution
- **Flexible Configuration**: YAML-based configuration for tasks, agents, and LLM parameters

## Architecture

### Core Components

```
AgentKernelArena/
├── main.py                      # Main orchestration entry point
├── config.yaml                  # Global configuration
├── src/
│   ├── module_registration.py  # Dynamic agent/prompt/post-processing loading
│   ├── preprocessing.py         # Workspace setup and environment checks
│   ├── prompt_builder.py        # Task prompt construction
│   ├── postprocessing.py        # Result analysis and report generation
│   ├── scoring.py               # Scoring logic for evaluation metrics
│   └── tasks.py                 # Task discovery and registration
├── agents/
│   ├── cursor/                  # Cursor agent integration
│   ├── claude_code/             # Claude Code agent integration
│   ├── single_llm_call/         # Single LLM call implementation
│   └── __init__.py              # Agent registry
└── tasks/                       # Task definitions
    ├── rocm-examples/           # ROCm example kernels
    ├── customer_hip/            # Custom HIP kernels
    ├── triton/                  # Triton benchmark kernels
    └── customer_pytorch/        # Custom PyTorch implementations
```

### Execution Flow

1. **Configuration Loading**: Load `config.yaml` with agent, task, and LLM settings
2. **Agent Registration**: Dynamically load agent launcher, prompt builder, and post-processing handler based on AgentType enum
3. **Task Discovery**: Scan `tasks/` directory for task configurations matching specified categories
4. **Workspace Setup**: Create isolated workspace with timestamp for each task
5. **Prompt Building**: Construct task-specific prompts from config, source code, and instructions
6. **Agent Execution**: Launch agent in workspace with constructed prompt
7. **Result Collection**: Save agent output, logs, and modified code
8. **Post-Processing**: Run compilation, correctness tests, performance profiling, and scoring
9. **Report Generation**: Generate comprehensive evaluation report with metrics

## Installation

### Prerequisites

- Python 3.8+
- ROCm toolkit (for HIP kernels): `hipcc`, `rocprof-compute`
- Triton (for Triton kernels)
- Git

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd AgentKernelArena

# Install dependencies
pip install -r requirements.txt

# Set up API keys (choose one or more)
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENROUTER_API_KEY="your_openrouter_key"

# Install agent CLIs (if using cursor or claude_code)
# For Claude Code:
npm install -g @anthropic-ai/claude-code

# For Cursor:
# Follow cursor-agent installation instructions
```

## Usage

### Basic Usage

1. **Configure `config.yaml`**:

```yaml
# Select agent type
agent:
  template: claude_code  # Options: cursor, claude_code, swe-agent, single_llm_call, geak
  max_iterations: 5

# Specify tasks to run
tasks:
  - rocm-examples/bitonic_sort
  - customer_hip/silu
  # - all  # Run ALL tasks

target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace

# Select LLM provider
provider: claude  # Options: openai, claude, openrouter, vllm
```

2. **Run evaluation**:

```bash
python main.py
```


### Advanced Usage

#### Running Specific Task Categories

```yaml
tasks:
  - rocm-examples/*           # All ROCm examples
  - customer_hip/mmcv/*       # All MMCV HIP kernels
  - triton/tritonbench/*      # All Triton benchmarks
```

## Task Configuration

Each task is defined by a `config.yaml` in its directory:

```yaml
# tasks/rocm-examples/bitonic_sort/config.yaml
source_file_path:
  - main.hip

target_kernel_functions:
  - bitonic_sort_kernel

compile_command:
  - make

correctness_command:
  - ./applications_bitonic_sort -l 15

performance_command:
  - rocprof-compute profile -n kernelgen --path rocprof_compute_profile --no-roof --join-type kernel -b SQ -b TCP -b TCC -- ./applications_bitonic_sort -l 15
  - rocprof-compute analyze --path rocprof_compute_profile -b 2
task_type: hip2hip
prompt:
  source_code: null      # Optional: override default source code inclusion
  instructions: null     # Optional: custom instructions
  cheatsheet: null       # Optional: provide cheatsheet/reference
```


## Scoring System

AgentKernelArena uses a cumulative scoring system:

| Metric | Points | Description |
|--------|--------|-------------|
| **Compilation** | 20 | Code compiles successfully without errors |
| **Correctness** | 100 | Code produces correct output (passes tests) |
| **Speedup** | ratio × 100 | Performance improvement over baseline |

**Example**: A submission that compiles (20), passes correctness (100), and achieves 1.5× speedup (150) would score 270 points.

Note: This is not the only way to score, but we have found it to be effective and helpful.

## Supported Agents

### Cursor
Interactive code editor agent with multi-turn conversation.

```yaml
agent:
  template: cursor
```

### Claude Code
Anthropic's official CLI agent for Claude.

```yaml
agent:
  template: claude_code
```

### SWE-agent
Software engineering agent for code modifications.

```yaml
agent:
  template: swe-agent
```

### OpenEvolve (GEAK)
Evolutionary coding agent for GPU kernel optimization using island-based evolution and MAP-Elites.

```yaml
agent:
  template: openevolve
```

**Configuration** (`agents/openevolve/agent_config.yaml`):
```yaml
llm:
  api_base: "https://api.openai.com/v1"  # Or custom endpoint
  models:
    - name: "claude-sonnet-4"
      weight: 1.0
max_iterations: 10
evaluator:
  timeout: 300  # seconds
  verbose: true
```

**Supported Task Types**:
- `instruction2triton`: Generate/optimize kernel from instructions
- `triton2triton`: Optimize existing Triton kernel

**See Also**: [INTEGRATION_openevolve.md](INTEGRATION_openevolve.md) for detailed documentation

## ROCmBench Triton Kernels

AgentKernelArena includes 31 ROCmBench tasks optimized for AMD ROCm GPUs, located in `tasks/triton/rocmbench/`.

### Available Kernels
Element-wise ops (add, mul, div), reductions (softmax, layernorm), matrix ops (gemm, matmul), and more.

### Testing OpenEvolve with ROCmBench

```bash
# Set environment
export OPENAI_API_KEY="your-api-key"
export ROCM_GOLDEN_DATA_PATH="/path/to/golden/results"

# Test single kernel
python main.py \
  --config config.yaml \
  --tasks triton/rocmbench/test_add_kernel

# Test multiple kernels
python main.py \
  --config config.yaml \
  --tasks triton/rocmbench/gemm triton/rocmbench/softmax
```

### Task Structure
```
tasks/triton/rocmbench/test_add_kernel/
├── config.yaml           # Task configuration
└── test_add_kernel.py    # Triton kernel with pytest tests
```

### Evaluation Commands
- **Correctness**: `pytest -vv -x test.py -k "not test_performance..."`
- **Performance**: `pytest -vv -x test.py -k "test_performance..."`

## Development

### Adding a New Agent

1. **Create agent directory**: `agents/your_agent/`

2. **Implement launch function**:

```python
# agents/your_agent/launch_agent.py
from agents import register_agent

@register_agent("your_agent")
def launch_agent(prompt: str, log_directory: str, workspace: str) -> str:
    """
    Launch your agent.

    Returns:
        str: Agent output
    """
    # Your agent implementation
    return result
```

3. **Register in module_registration.py**:

```python
# Add to AgentType enum
class AgentType(Enum):
    YOUR_AGENT = "your_agent"

# Add import in load_agent_launcher
if agent_type == AgentType.YOUR_AGENT:
    from agents.your_agent import launch_agent
```

4. **Add prompt builder support** (if needed):

```python
# In load_prompt_builder
if agent_type in [..., AgentType.YOUR_AGENT]:
    return prompt_builder
```

5. **Add post-processing support** (if needed):

```python
# In load_post_processing_handler
if agent_type in [..., AgentType.YOUR_AGENT]:
    return general_post_processing
```

### Adding a New Task

1. **Create task directory**: `tasks/category/task_name/`

2. **Add source files**: `main.hip`, `Makefile`, etc.

3. **Create config.yaml**:

```yaml
source_file_path:
  - main.hip

target_kernel_functions:
  - your_kernel_function

compile_command:
  - make

correctness_command:
  - ./your_executable --test

performance_command:
  - rocprof-compute profile ... -- ./your_executable
  - rocprof-compute analyze ...
task_type: triton2triton

prompt:
  source_code: null
  instructions: null
  cheatsheet: null
```

4. **Add baseline performance** (optional): Create `baseline.txt` with expected performance metrics

### Type Safety

All functions in the codebase are strongly typed. When adding new code, ensure:

```python
from typing import Callable, Dict, Optional, List
from pathlib import Path
import logging

def your_function(
    param1: str,
    param2: Path,
    logger: logging.Logger,
    optional_param: Optional[str] = None
) -> Dict[str, str]:
    """
    Function description.

    Args:
        param1: Description
        param2: Description
        logger: Logger instance
        optional_param: Optional description

    Returns:
        Description of return value
    """
    # Implementation
    pass
```

