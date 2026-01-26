# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import yaml
import logging
from pathlib import Path
from src.prompts import task_type


def source_code(config: dict) -> str:
    """Generate a comprehensive task definition prompt for HIP kernel optimization."""
    
    # Format lists properly
    source_files = "\n    - ".join(config['source_file_path']) if isinstance(config['source_file_path'], list) else config['source_file_path']
    target_kernels = "\n    - ".join(config['target_kernel_functions']) if isinstance(config['target_kernel_functions'], list) else config['target_kernel_functions']
    
    return f"""
### Source Code
**File(s) to optimize:**
    - {source_files}

**Target kernel function(s):**
    - {target_kernels}
"""


def instructions(config: dict) -> str:

    # Handle multiple compile/test commands
    compile_cmds = "\n    ".join(config['compile_command']) if isinstance(config['compile_command'], list) else config['compile_command']
    correctness_cmds = "\n    ".join(config['correctness_command']) if isinstance(config['correctness_command'], list) else config['correctness_command']
    performance_cmds = "\n    ".join(config.get('performance_command', ['Not specified'])) if isinstance(config.get('performance_command'), list) else config.get('performance_command', 'Not specified')

    return f"""
### Instructions

1. **Compilation Phase**
    Execute the following command(s) to build the optimized code:
    {compile_cmds}
    - Compilation must succeed without errors or warnings.

2. **Correctness Validation Phase** (MANDATORY)
    Execute the following command(s) to verify correctness:
    {correctness_cmds}
    - The output must indicate "Validation passed" or equivalent success message.
    - Any validation errors are unacceptable - the optimization must produce identical results to the original.
    - Compare kernel execution time before and after optimization.

3. **Performance Analysis Phase** (OPTIONAL)
    Execute the following command(s) to measure performance:
    {performance_cmds}
    - Profiling could help you to identify the bottlenecks and optimize the kernel.
"""


def task_result_format(task_result_template: str) -> str:
    """
    Generate the output format section by reading the task result template.

    Returns:
        str: Formatted output requirements section with the YAML template
    """
    # Load the task result template
    if task_result_template == None:
        template_path = Path(__file__).parent / "prompts/task_result_template.yaml"
    else:
        template_path = Path(__file__).parent / "prompts" / task_result_template

    with open(template_path, 'r') as f:
        template_content = f.read()

    return f"""
### Output Format

**IMPORTANT**: After completing the optimization task, you MUST fill out the following YAML template with your results and save it as `task_result.yaml` in the workspace directory.

```yaml
{template_content}```

Instructions for filling the template: Replace all placeholder values with actual results from your optimization process
Template location: Save the completed template as `task_result.yaml` in your working directory.

"""


def prompt_builder(task_config_dir: str, workspace_directory: Path, eval_config: dict, logger: logging.Logger) -> str:
    """
    Build the initial prompt for the agent based on task configuration.

    Args:
        task_config_dir: Path to the task's config.yaml
        workspace_directory: Path to the duplicated workspace for the agent
        task_type_name: Type of task (hip2hip, pytorch2hip, triton2triton)
        logger: Logger instance

    Returns:
        str: The complete prompt for the agent
    """
    # Load task configuration
    task_config_path = Path(task_config_dir)
    with open(task_config_path, 'r') as f:
        task_config = yaml.safe_load(f)

    task_type_name = task_config.get('task_type')
    target_gpu_model = eval_config.get('target_gpu_model', 'MI300')
    logger.info(f"Building prompt from config: {task_config_path}")

    # Build prompt sections
    prompt_sections = []

    # 1. Task Type Section
    if not task_type_name:
        raise ValueError("task_type is missing in task config")
    if task_type_name == 'hip2hip':
        task_type_prompt = task_type.hip2hip_task_type()
    elif task_type_name == 'pytorch2hip':
        task_type_prompt = task_type.pytorch2hip_task_type()
    elif task_type_name == 'triton2triton':
        task_type_prompt = task_type.triton2triton_task_type()
    elif task_type_name == 'cuda2hip':
        task_type_prompt = task_type.cuda2hip_task_type()
    elif task_type_name == 'instruction2triton':
        task_type_prompt = task_type.instruction2triton_task_type()
    else:
        raise ValueError(f"Unknown task type: {task_type_name}")

    prompt_sections.append(task_type_prompt)
    
    # 2. Source Code Section
    source_code_prompt = task_config.get('prompt', {}).get('source_code')
    source_file_path = task_config.get('source_file_path', [])
    if source_code_prompt is None:
        if any(x is not None for x in source_file_path):
            source_code_prompt = source_code(task_config)
        else:
            source_code_prompt = ""

    prompt_sections.append(source_code_prompt)

    # 3. Instructions Section
    instructions_prompt = task_config.get('prompt', {}).get('instructions')
    if instructions_prompt is None:
        instructions_prompt = instructions(task_config)

    prompt_sections.append(instructions_prompt)

    # 4. Output Format Section
    task_result_template = task_config.get('task_result_template', '')
    task_result_prompt = task_config.get('prompt', {}).get('output_format')
    if task_result_prompt is None:
        task_result_prompt = task_result_format(task_result_template)

    prompt_sections.append(task_result_prompt)

    # 5. Cheatsheet Section
    cheatsheet_prompt = task_config.get('prompt', {}).get('cheatsheet')
    if cheatsheet_prompt is None:
        # Load cheatsheet mapping and pick by target language (task_type suffix after '2')
        try:
            project_root = Path(__file__).resolve().parent.parent
            cheatsheet_config_path = project_root / "src/prompts/cheatsheet/default_cheatsheet.yaml"
            cheatsheet_config = yaml.safe_load(cheatsheet_config_path.read_text()) or {}
            target_language = (task_type_name.split('2')[-1] if '2' in task_type_name else task_type_name).lower()
            gpu_key = str(target_gpu_model)
            gpu_map = (
                cheatsheet_config.get(gpu_key)
                or cheatsheet_config.get(gpu_key.upper())
                or cheatsheet_config.get(gpu_key.lower())
            )
            cheatsheet_file = gpu_map.get(target_language) if isinstance(gpu_map, dict) else None
            if cheatsheet_file:
                cheatsheet_path = project_root / cheatsheet_file
                cheatsheet_prompt = cheatsheet_path.read_text()
                logger.info(f"Loaded cheatsheet for target '{target_language}' on GPU '{target_gpu_model}': {cheatsheet_path}")
            else:
                logger.warning(f"No cheatsheet mapping for target '{target_language}' on GPU '{target_gpu_model}', using empty cheatsheet")
                cheatsheet_prompt = ""
        except Exception as e:
            logger.warning(f"Failed to load cheatsheet from mapping: {e}, using empty cheatsheet")
            cheatsheet_prompt = ""

    prompt_sections.append(cheatsheet_prompt)

    # 6. Workspace Directory Information
    workspace_info = f"""
### Workspace Directory
Your working directory is: `{workspace_directory}`

This duplicated workspace contains:
- All source files that need to be optimized
- Build system and compilation tools
- Test and validation scripts
- Profiling tools

You can directly modify the kernel code in this workspace. All changes should be made within this directory.

"""
    prompt_sections.append(workspace_info)

    # Combine all sections
    final_prompt = "\n\n".join(prompt_sections)

    logger.info(f"Prompt built successfully, total length: {len(final_prompt)} characters")

    return final_prompt
