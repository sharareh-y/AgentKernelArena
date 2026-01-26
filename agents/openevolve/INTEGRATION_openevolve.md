# OpenEvolve Integration - Documentation

## Overview

This document describes the integration of the OpenEvolve (GEAK) evolutionary coding agent into AgentKernelArena for GPU kernel optimization.

## Integration Architecture

### Core Components

**1. Agent Launcher** (`agents/openevolve/launch_agent.py`)
- Adapts AgentKernelArena task structure to OpenEvolve requirements
- Creates dynamic evaluator from task configuration
- Manages workspace and file paths
- Handles both `instruction2triton` and `triton2triton` task types

**2. Agent Configuration** (`agents/openevolve/agent_config.yaml`)
- LLM settings (model, API endpoint, temperature)
- Evolution parameters (iterations, population size)
- Evaluator settings (timeout, parallel evaluations, verbose mode)

**3. Agent Registration** (`src/module_registration.py`)
- Added `AgentType.OPENEVOLVE` enum
- Integrated with existing prompt building and post-processing pipeline

## Configuration

### Global Config (`config.yaml`)

```yaml
agent:
  template: openevolve

tasks:
  - triton/rocmbench/test_add_kernel

task_type: instruction2triton
target_gpu_model: MI300
```

### Agent Config (`agents/openevolve/agent_config.yaml`)

```yaml
llm:
  api_base: "https://api.openai.com/v1"  # Or custom endpoint
  models:
    - name: "claude-sonnet-4"
      weight: 1.0
  temperature: 0.7

max_iterations: 10
population_size: 20

database:
  path: "programs.db"

evaluator:
  timeout: 300  # seconds
  parallel_evaluations: 1
  cascade_evaluation: false
  verbose: true
```

### Task Config Example

```yaml
task_type: instruction2triton
source_file_path: null

target_kernel_functions:
  - add_kernel

compile_command:
  - python -c "import ast; ast.parse(open('test_add_kernel.py').read())"

correctness_command:
  - pytest -vv -x --maxfail=1 test_add_kernel.py -k "not test_performance and not test_save_performance_results"

performance_command:
  - pytest -vv -x --maxfail=1 test_add_kernel.py -k "test_performance or test_save_performance_results"

prompt:
  task: "Optimize Triton kernel for AMD GPU"
  instructions: "Improve performance while maintaining correctness"
```

## Usage

### Basic Usage

```bash
# Set environment
export OPENAI_API_KEY="your-api-key"
export ROCM_GOLDEN_DATA_PATH="/path/to/golden/results"

# Run single task
python main.py --tasks triton/rocmbench/test_add_kernel

# Run multiple tasks
python main.py --tasks triton/rocmbench/gemm triton/rocmbench/softmax
```
## 📝 Kernel Structure Format

GEAK-OpenEvolve requires initial programs (starting kernels) to follow a specific structure for ROCm evaluation. The kernel file must contain three main sections separated by a special separator:

### File Structure

```python
# ============================================================
# SECTION 1: Triton Kernel Code
# ============================================================
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
  
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


# ============================================================
# SECTION 2: SEPARATOR (146 '#' characters)
# ============================================================
##################################################################################################################################################


# ============================================================
# SECTION 3: Benchmarking & Testing Code (pytest)
# ============================================================
import numpy as np
import random
import torch 
import os
from numpy.random import RandomState
import pytest
from torch.testing import assert_close
from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict

import triton
import triton.language as tl

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
}

result_gold = {}

######################################## HELPERS for Eval ######################################## 
# Helper function to define GB/s for add_kernel
def calculate_add_gbps(params: Dict, ms: float) -> float:
    # params will contain 'SIZE', 'dtype_str'
    size = params['SIZE']
    dtype = dtype_mapping[params['dtype_str']]
    # For add: read x, read y, write output
    # If x, y, output are torch.Tensor objects passed to this calculator:
    # total_bytes = (x.numel() * x.element_size() +
    #                y.numel() * y.element_size() +
    #                output.numel() * output.element_size())
    # If only params are available:
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    total_bytes = 3 * size * bytes_per_element # 2 reads, 1 write
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

# Helper function to define TFLOPS for add_kernel
def calculate_add_tflops(params: Dict, ms: float) -> float:
    size = params['SIZE']
    # For add: N operations (N additions)
    flops = size
    tflops = flops / (ms / 1000) / 1e12
    return tflops

def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across multiple libraries and configure PyTorch for deterministic behavior.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch on all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

######################################## HELPERS for Eval ######################################## 




@pytest.mark.parametrize('SIZE,BLOCK_SIZE,dtype_str',
                         [(98432, 1024, dtype_str) for dtype_str in ['float16', 'float32']])
def test_add(SIZE, BLOCK_SIZE, dtype_str, request):
    set_seed()

    dtype = dtype_mapping[dtype_str]
    output = torch.empty(SIZE, device='cuda', dtype=dtype)
    x = torch.randn(SIZE, device='cuda', dtype=dtype)
    y = torch.randn(SIZE, device='cuda', dtype=dtype)

    def grid(meta):
        return (triton.cdiv(SIZE, meta['BLOCK_SIZE']), )

    add_kernel[grid](x, y, output, SIZE, BLOCK_SIZE=BLOCK_SIZE)

    output_torch = x + y
    torch.set_printoptions(profile='full')

    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    ################### save tri_out in result_gold ###################
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = output.clone().detach().cpu()
    ################################################################### 

    assert_close(output, output_torch, rtol=1e-2, atol=1e-3, check_dtype=False)


OP_NAME_FOR_BENCHMARK = "add_kernel_perf"

@pytest.mark.parametrize('SIZE,BLOCK_SIZE_ARG,dtype_str', # BLOCK_SIZE_ARG is the pytest param name
                         [(98432, 1024, dtype_str) for dtype_str in ['float16', 'float32']] +
                         [(1048576, 2048, dtype_str) for dtype_str in ['float16', 'float32']]
                        )
def test_performance(SIZE, BLOCK_SIZE_ARG, dtype_str, request): # Function accepts BLOCK_SIZE_ARG
    set_seed()
    dtype = dtype_mapping[dtype_str]
    x = torch.randn(SIZE, device='cuda', dtype=dtype)
    y = torch.randn(SIZE, device='cuda', dtype=dtype)
    output = torch.empty(SIZE, device='cuda', dtype=dtype)

    # Kernel launch grid
    # The 'meta' dict passed to the grid lambda by Triton contains the constexpr arguments
    # that were passed to the kernel launch.
    # When we call `add_kernel[grid](..., BLOCK_SIZE=BLOCK_SIZE_ARG)`,
    # the `meta` dict will have a key 'BLOCK_SIZE' (the name of the constexpr in the kernel signature)
    # and its value will be the runtime `BLOCK_SIZE_ARG`.
    grid = lambda meta: (triton.cdiv(SIZE, meta['BLOCK_SIZE']),) # ***** CORRECTED HERE *****

    kernel_args = [x, y, output, SIZE]
    
    # The op_lambda passes BLOCK_SIZE_ARG (runtime value) as the kernel's `BLOCK_SIZE` (constexpr name)
    op_lambda = lambda: add_kernel[grid](*kernel_args, BLOCK_SIZE=BLOCK_SIZE_ARG)

    bench_config = do_bench_config(warm_up=25, repetition=100) # Smaller for faster debug
    benchmarker = PytestBenchmarker(op_callable=op_lambda,
                                    op_name=OP_NAME_FOR_BENCHMARK,
                                    config=bench_config)

    # The dictionary passed to calculators should use consistent keys
    current_params_for_calculators = {"SIZE": SIZE, "BLOCK_SIZE_RUNTIME": BLOCK_SIZE_ARG, "dtype_str": dtype_str}
    # Note: I used "BLOCK_SIZE_RUNTIME" here to be explicit that it's the value from parametrize,
    # not necessarily the same as the constexpr name if they differed.
    # If your calculators expect 'BLOCK_SIZE', then use that:
    # current_params_for_calculators = {"SIZE": SIZE, "BLOCK_SIZE": BLOCK_SIZE_ARG, "dtype_str": dtype_str}


    benchmarker.run_benchmark(current_params_dict=current_params_for_calculators,
                              gbps_calculator=calculate_add_gbps,
                              tflops_calculator=calculate_add_tflops)
    
######################################## HELPERS for Eval ########################################     
# --- Pytest hook to save the dictionary at the end of the session ---  
def test_save_results():  
    """  
    Called after whole test run finished, right before returning the exit status to the system.  
    """
    print('Inside session finish...')
    if "_CALL_SUCCESS_" not in result_gold:
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[0.0]])
    OUTPUT_FILENAME = __file__.replace('.','_') + '.pt'
    print(f"\nSaving all y_triton results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} y_triton tensors to {OUTPUT_FILENAME}.")  


def test_save_performance_results():
    """
    Called after the test_performance function finishes.
    This is a separate hook to ensure performance results are saved.
    """
    print('\nPytest session finishing... Saving benchmark results...')

    output_directory = os.path.join(os.path.dirname(__file__), "perf")  # Save in a "perf" subdirectory next to the test file
    os.makedirs(output_directory, exist_ok=True)
    
    save_all_benchmark_results(output_directory)
    print(f"All benchmark results attempted to save to: {output_directory}")


######################################## HELPERS for Eval ########################################
```


## ROCmBench Tasks

31 Triton kernels optimized for AMD ROCm GPUs are available in `tasks/triton/rocmbench/`:

### Task Structure
```
tasks/triton/rocmbench/test_add_kernel/
├── config.yaml           # Task configuration
└── test_add_kernel.py    # Kernel with pytest tests
```

## Adding New Kernels

### Step 1: Create Task Directory
```bash
mkdir -p tasks/triton/custom/my_kernel
```

### Step 2: Add Kernel File
```python
# tasks/triton/custom/my_kernel/my_kernel.py
import triton
import triton.language as tl

@triton.jit
def my_kernel(...):
    # Kernel implementation
    pass

# Pytest tests
@pytest.mark.parametrize(...)
def test_my_kernel(...):
    # Test implementation
    pass

def test_performance(...):
    # Performance benchmarking
    pass
```

### Step 3: Create Configuration
```yaml
# tasks/triton/custom/my_kernel/config.yaml
task_type: instruction2triton  # or triton2triton

target_kernel_functions:
  - my_kernel

compile_command:
  - python -c "import ast; ast.parse(open('my_kernel.py').read())"

correctness_command:
  - pytest -vv -x --maxfail=1 my_kernel.py -k "not test_performance..."

performance_command:
  - pytest -vv -x --maxfail=1 my_kernel.py -k "test_performance..."

prompt:
  task: "Your task description"
  instructions: "Your specific instructions"
```

### Step 4: Run
```bash
python main.py --tasks triton/custom/my_kernel
```

**No code changes needed!** The integration is fully generic.

## External Dependencies

### geak-openevolve

GitHub Repository: [AMD-AGI/GEAK-agent (geak-openevolve branch)](https://github.com/AMD-AGI/GEAK-agent/tree/geak-openevolve)

**Status**: ✅ Compatible

The OpenEvolve evolutionary coding agent integrated into AgentKernelArena. Requires environment-aware API key handling (already implemented).

### GEAK-eval-OE

GitHub Repository: [AMD-AGI/GEAK-eval (openevolve branch)](https://github.com/AMD-AGI/GEAK-eval/tree/openevolve)

**Status**: ✅ No changes needed

Used for golden performance data and pytest-based evaluation utilities. Works as-is with the integration.

## Implementation Details

### How Evaluation Works

1. **Workspace Setup**: Task files copied to isolated workspace
2. **Initial Evaluation**: Original kernel evaluated for baseline
3. **Evolution**: OpenEvolve generates improved variants
4. **Dynamic Evaluation**: Each variant evaluated with:
   - Compile: Syntax check
   - Correctness: Pytest with specific test filter
   - Performance: Pytest with performance test filter
5. **Selection**: Best variants kept for next generation
6. **Iteration**: Process repeats for `max_iterations`

### Evaluator Creation

```python
def create_evaluator(task_config, workspace):
    """Creates evaluator from task configuration"""
    compile_cmds = task_config['compile_command']
    correctness_cmds = task_config['correctness_command']
    performance_cmds = task_config['performance_command']
    
    def evaluate(program_text, ...):
        # Run commands with evolved program
        # Return success/failure metrics
        pass
    
    return evaluate
```

This pattern ensures the integration works with ANY kernel that follows the task configuration format.

## Performance Metrics

OpenEvolve tracks:
- **Compilation Success**: Binary pass/fail
- **Correctness Score**: Based on test pass rate
- **Performance**: Execution time, GB/s, TFLOPS
- **Combined Score**: Weighted combination

## Limitations

1. **Pytest Dependency**: Assumes tests use pytest
   - Extensible to other frameworks if needed

2. **Python Files**: Assumes kernels are `.py` files
   - Can be extended for other extensions

3. **Shell Commands**: Assumes commands are shell strings
   - Already supports list-of-strings format

