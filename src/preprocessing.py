# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# This script will setup environment tools and dependencies. It will also provide duplicated workspace for the agent
import os
import shutil
import logging
from pathlib import Path


def check_environment() -> None:
    # check hipcc, rocprof-compute
    if "hipcc" not in os.environ["PATH"]:
        raise ValueError("hipcc is not in the PATH")
    if "rocprof-compute" not in os.environ["PATH"]:
        raise ValueError("rocprof-compute is not in the PATH")
    pass


def setup_workspace(task_config_dir: str, workspace_directory: str, timestamp: str, logger: logging.Logger) -> Path:
    """
    Setup workspace for agent execution by duplicating task directory.

    Args:
        task_config_dir: Path to task's config.yaml
        workspace_directory: Base workspace directory
        timestamp: Timestamp string for unique workspace naming
        logger: Logger instance

    Returns:
        Path to the created workspace directory
    """
    # 1. Get task_folder name (parent directory of task_config_dir)
    task_config_path = Path(task_config_dir)
    task_folder = task_config_path.parent
    task_folder_name = task_folder.name

    # 2. Create new directory with timestamp suffix under workspace_dir
    new_folder_name = f"{task_folder_name}_{timestamp}"
    workspace_path = Path(workspace_directory) / new_folder_name
    workspace_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created workspace directory: {workspace_path}")

    # 3. Duplicate all content under task_folder to the new workspace folder
    for item in task_folder.iterdir():
        src = item
        dst = workspace_path / item.name
        if item.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    logger.info(f"Copied task folder content from {task_folder} to {workspace_path}")

    return workspace_path
