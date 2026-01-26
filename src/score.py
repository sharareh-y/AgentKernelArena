# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import yaml
from pathlib import Path

def score(pass_compilation: bool, pass_correctness: bool, base_execution_time: float, best_optimized_execution_time: float) -> float:
    """
    Calculate the optimization task score based on compilation, correctness, and performance.

    Scoring rules:
    - Pass compilation: +20 points
    - Pass correctness: +100 points
    - Speedup (only if both compilation and correctness pass): speedup_ratio * 100 points

    Args:
        pass_compilation: Whether compilation succeeded
        pass_correctness: Whether correctness tests passed
        base_execution_time: Baseline execution time (must be > 0)
        best_optimized_execution_time: Optimized execution time (must be > 0)

    Returns:
        float: Total score
            - 0: Compilation failed
            - 20: Compilation passed, correctness failed
            - 120+: Both passed, 120 base + speedup_ratio * 100
    """
    total_score = 0.0

    # 1. Compilation check: +20 points
    if not pass_compilation:
        return 0.0

    total_score += 20.0

    # 2. Correctness check: +100 points
    if not pass_correctness:
        return total_score

    total_score += 100.0

    # 3. Performance speedup: speedup_ratio * 100 (only if both compilation and correctness passed)
    if base_execution_time > 0 and best_optimized_execution_time > 0:
        speedup_ratio = base_execution_time / best_optimized_execution_time
        total_score += speedup_ratio * 100.0

    return total_score


def task_result_scoring(workspace_path: str) -> float:
    """
    Read task_result.yaml from workspace, calculate score, and append it to the file.

    Args:
        workspace_path: Path to the workspace directory containing task_result.yaml

    Returns:
        float: Calculated score

    Raises:
        FileNotFoundError: If task_result.yaml doesn't exist
        KeyError: If required fields are missing from the YAML
    """
    workspace = Path(workspace_path)
    result_file = workspace / "task_result.yaml"

    # Check if file exists
    if not result_file.exists():
        raise FileNotFoundError(f"task_result.yaml not found in {workspace_path}")

    # Read the YAML file
    with open(result_file, 'r') as f:
        result_data = yaml.safe_load(f)

    # Extract required fields
    pass_compilation = result_data.get('pass_compilation', False)
    pass_correctness = result_data.get('pass_correctness', False)
    base_execution_time = result_data.get('base_execution_time', 0.0)
    best_optimized_execution_time = result_data.get('best_optimized_execution_time', 0.0)

    # Calculate score
    calculated_score = score(
        pass_compilation=pass_compilation,
        pass_correctness=pass_correctness,
        base_execution_time=base_execution_time,
        best_optimized_execution_time=best_optimized_execution_time
    )

    # Add score to the data
    result_data['score'] = calculated_score

    # Write back to the YAML file
    with open(result_file, 'w') as f:
        yaml.dump(result_data, f, default_flow_style=False, sort_keys=False)

    return calculated_score