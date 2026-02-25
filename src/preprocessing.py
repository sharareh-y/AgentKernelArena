# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# This script will setup environment tools and dependencies. It will also provide duplicated workspace for the agent
import os
import shutil
import logging
from pathlib import Path

import yaml


def _resolve_gfx_arch(target_gpu_model: str) -> str | None:
    """
    Look up the gfx architecture token (e.g. 'gfx942') for a given GPU model
    name (e.g. 'MI300') from default_cheatsheet.yaml.

    Returns None if the GPU model is not found.
    """
    cheatsheet_path = (
        Path(__file__).resolve().parent / "prompts" / "cheatsheet" / "default_cheatsheet.yaml"
    )
    try:
        config = yaml.safe_load(cheatsheet_path.read_text()) or {}
    except Exception:
        return None

    arch_map = config.get("architecture", {})
    gpu_key = str(target_gpu_model)
    entry = (
        arch_map.get(gpu_key)
        or arch_map.get(gpu_key.upper())
        or arch_map.get(gpu_key.lower())
    )
    if isinstance(entry, dict):
        return entry.get("gfx_arch")
    return None


def setup_rocm_env(target_gpu_model: str, logger: logging.Logger) -> None:
    """
    Set PYTORCH_ROCM_ARCH (and related env vars) based on config.yaml's
    target_gpu_model so that torch.utils.cpp_extension.load() and hipcc
    compile for the correct GPU architecture.

    Should be called once at the start of main(), before any task is launched.
    """
    gfx_arch = _resolve_gfx_arch(target_gpu_model)
    if not gfx_arch:
        logger.warning(
            f"Could not resolve gfx arch for GPU model '{target_gpu_model}'. "
            "PYTORCH_ROCM_ARCH will not be set; PyTorch will fall back to its built-in arch list."
        )
        return

    os.environ["PYTORCH_ROCM_ARCH"] = gfx_arch
    logger.info(f"Set PYTORCH_ROCM_ARCH={gfx_arch} (from target_gpu_model={target_gpu_model})")


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
