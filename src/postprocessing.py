# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any
from src.score import task_result_scoring



def general_log_report(aggregate_result: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Log a formatted report using the provided logger.

    Args:
        aggregate_result: Report dictionary from post_processing()
        logger: Logger instance to use for output
    """
    logger.info("=" * 80)
    logger.info("AgentKernelArena Task Results Report")
    logger.info("=" * 80)

    # Overall statistics
    logger.info("Overall Statistics:")
    logger.info(f"  Total Tasks:           {aggregate_result['total_tasks']}")
    logger.info(f"  Total Score:           {aggregate_result['total_score']:.2f}")
    logger.info(f"  Average Score:         {aggregate_result['average_score']:.2f}")

    # Compilation statistics
    logger.info("Compilation:")
    logger.info(f"  Pass Count:            {aggregate_result['compilation_pass_count']}/{aggregate_result['total_tasks']}")
    logger.info(f"  Pass Rate:             {aggregate_result['compilation_pass_rate']:.1f}%")

    # Correctness statistics
    logger.info("Correctness:")
    logger.info(f"  Pass Count:            {aggregate_result['correctness_pass_count']}/{aggregate_result['total_tasks']}")
    logger.info(f"  Pass Rate:             {aggregate_result['correctness_pass_rate']:.1f}%")

    # Performance statistics
    logger.info("Performance:")
    logger.info(f"  Speedup > 1.0 Count:   {aggregate_result['speedup_gt_1_count']}/{aggregate_result['total_tasks']}")
    logger.info(f"  Speedup > 1.0 Rate:    {aggregate_result['speedup_gt_1_rate']:.1f}%")
    logger.info(f"  Average Speedup:       {aggregate_result['average_speedup']:.2f}x")
    logger.info(f"  Valid Speedup Count:   {aggregate_result['valid_speedup_count']}")

    # Task details
    logger.info("Task Details:")
    logger.info("-" * 80)

    for task in aggregate_result['task_details']:
        status = "PASS" if task['pass_correctness'] else ("PARTIAL" if task['pass_compilation'] else "FAIL")
        logger.info(f"{status:<8} {task['task_name']:<40} Score: {task['score']:>6.1f}  Speedup: {task['speedup_ratio']:.2f}x")
        if task['error']:
            logger.info(f"         Error: {task['error']}")

    logger.info("=" * 80)


def general_post_processing(workspace_paths: List[str], logger: logging.Logger) -> None:
    """
    Process all task results and generate a comprehensive report.

    Args:
        workspace_paths: List of workspace directory paths
        logger: Logger instance to use for output
    Returns:
        dict: Report containing statistics and task details
            - total_tasks: Total number of tasks
            - total_score: Sum of all scores
            - compilation_pass_count: Number of tasks that passed compilation
            - compilation_pass_rate: Percentage of tasks that passed compilation
            - correctness_pass_count: Number of tasks that passed correctness tests
            - correctness_pass_rate: Percentage of tasks that passed correctness tests
            - speedup_gt_1_count: Number of tasks with speedup > 1.0
            - speedup_gt_1_rate: Percentage of tasks with speedup > 1.0
            - average_speedup: Average speedup ratio (only valid speedups)
            - task_details: List of detailed information for each task
    """
    total_tasks = len(workspace_paths)
    total_score = 0.0
    compilation_pass_count = 0
    correctness_pass_count = 0
    speedup_gt_1_count = 0
    speedup_values = []

    task_details = []

    for workspace_path in workspace_paths:
        workspace = Path(workspace_path)
        task_name = workspace.name  # Get task folder name

        task_info = {
            'task_name': task_name,
            'workspace_path': str(workspace_path),
            'score': 0.0,
            'pass_compilation': False,
            'pass_correctness': False,
            'speedup_ratio': 0.0,
            'error': None
        }

        try:
            # Try to calculate score (this will read task_result.yaml and update it with score)
            calculated_score = task_result_scoring(str(workspace_path))
            task_info['score'] = calculated_score

            # Read the task_result.yaml to get detailed information
            result_file = workspace / "task_result.yaml"
            with open(result_file, 'r') as f:
                result_data = yaml.safe_load(f)

            # Extract information
            task_info['task_name'] = result_data.get('task_name', task_name)
            task_info['pass_compilation'] = result_data.get('pass_compilation', False)
            task_info['pass_correctness'] = result_data.get('pass_correctness', False)

            base_execution_time = result_data.get('base_execution_time', 0.0)
            best_optimized_execution_time = result_data.get('best_optimized_execution_time', 0.0)
            if base_execution_time > 0 and best_optimized_execution_time > 0:
                # TODO: remove speedup_ratio field from task_result.yaml in future versions
                task_info['speedup_ratio'] = base_execution_time / best_optimized_execution_time
            else:
                task_info['speedup_ratio'] = result_data.get('speedup_ratio', 0.0)

            # Update counters
            total_score += calculated_score

            if task_info['pass_compilation']:
                compilation_pass_count += 1

            if task_info['pass_correctness']:
                correctness_pass_count += 1

            # Check speedup (only if both compilation and correctness passed)
            if task_info['pass_compilation'] and task_info['pass_correctness']:
                speedup = task_info['speedup_ratio']
                if speedup > 1.0:
                    speedup_gt_1_count += 1
                if speedup > 0:  # Only include valid speedups
                    speedup_values.append(speedup)

        except FileNotFoundError as e:
            task_info['error'] = f"task_result.yaml not found: {e}"
            task_info['score'] = 0.0

        except (KeyError, ValueError, TypeError) as e:
            task_info['error'] = f"Invalid or missing data in task_result.yaml: {e}"
            task_info['score'] = 0.0

        except Exception as e:
            task_info['error'] = f"Unexpected error: {e}"
            task_info['score'] = 0.0

        task_details.append(task_info)

    # Calculate rates
    compilation_pass_rate = (compilation_pass_count / total_tasks * 100) if total_tasks > 0 else 0.0
    correctness_pass_rate = (correctness_pass_count / total_tasks * 100) if total_tasks > 0 else 0.0
    speedup_gt_1_rate = (speedup_gt_1_count / total_tasks * 100) if total_tasks > 0 else 0.0

    # Calculate average speedup (only for valid speedups)
    average_speedup = (sum(speedup_values) / len(speedup_values)) if speedup_values else 0.0

    # Generate aggregate_result
    aggregate_result = {
        'total_tasks': total_tasks,
        'total_score': total_score,
        'average_score': total_score / total_tasks if total_tasks > 0 else 0.0,

        'compilation_pass_count': compilation_pass_count,
        'compilation_pass_rate': compilation_pass_rate,

        'correctness_pass_count': correctness_pass_count,
        'correctness_pass_rate': correctness_pass_rate,

        'speedup_gt_1_count': speedup_gt_1_count,
        'speedup_gt_1_rate': speedup_gt_1_rate,

        'average_speedup': average_speedup,
        'valid_speedup_count': len(speedup_values),

        'task_details': task_details
    }

    general_log_report(aggregate_result, logger)