# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from src.tasks import get_task_config
from src.preprocessing import setup_workspace, setup_rocm_env
from src.module_registration import AgentType, load_agent_launcher, load_post_processing_handler


parser = argparse.ArgumentParser(description="arguments for AgentKernelArena")
parser.add_argument("--config_name", type=str, default="config.yaml",help="the config of AgentKernelArena, default set to config. \
                    You can set different tasks in different config yaml file in order to run multi evaluation task in one folder.")

def main() -> None:
    """Main entry point for AgentKernelArena framework."""
    args = parser.parse_args()

    # Load config.yaml
    with open(args.config_name, 'r') as f:
        config = yaml.safe_load(f)

    # Extract configuration
    tasks = config['tasks']  # Now directly a list
    agent_string = config['agent']['template']
    target_gpu_model = config['target_gpu_model']

    log_directory = config['log_directory']
    workspace_directory_prefix = config['workspace_directory_prefix']

    # Convert agent string to AgentType enum
    try:
        agent = AgentType.from_string(agent_string)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Build workspace directory name
    workspace_directory_name = f"{workspace_directory_prefix}_{target_gpu_model}_{agent.value}"    
    project_root = Path(__file__).resolve().parent
    workspace_directory = (project_root / workspace_directory_name).resolve()

    # Create log file with target_gpu_model, agent, and timestamp    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(log_directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"{target_gpu_model}_{agent.value}_{timestamp}.log"
    log_path = log_dir / log_filename

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("AgentKernelArena Framework Started")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_path}")
    logger.info(f"Agent: {agent.value}")
    logger.info(f"Target Architecture: {target_gpu_model}")
    logger.info(f"Workspace Directory: {workspace_directory}")

    # Set PYTORCH_ROCM_ARCH based on target_gpu_model before any task runs
    setup_rocm_env(target_gpu_model, logger)

    # Load agent launcher
    try:
        agent_launcher = load_agent_launcher(agent, logger)
    except Exception as e:
        logger.error(f"Failed to load agent launcher: {e}")
        return


    # Get task config
    if 'all' in tasks:
        task_config_dict = get_task_config()
    else:
        task_config_dict = {}
        for category in tasks:
            task_config_dict.update(get_task_config(category=category))

    logger.info(f"Found {len(task_config_dict)} tasks to execute")
    logger.info(f"Tasks: {list(task_config_dict.keys())}")

    # Collect workspace paths for post-processing
    workspace_paths = []

    # Run tasks
    for idx, (task_name, task_config_dir) in enumerate(task_config_dict.items(), 1):
        logger.info("=" * 80)
        logger.info(f"Task {idx}/{len(task_config_dict)}: {task_name}")
        logger.info("=" * 80)
        
        try:
            # Setup workspace
            workspace_path = setup_workspace(task_config_dir, workspace_directory, timestamp, logger)
            
            # Launch agent
            logger.info(f"Launching agent: {agent.value}")

            # For agentic approaches (cursor, claude_code, etc.)
            result = agent_launcher(
                eval_config=config,
                task_config_dir=task_config_dir,
                workspace=str(workspace_path)
            )

            logger.info(f"Agent execution completed")
            logger.info(f"Task {task_name} completed successfully")

            # Add workspace path to list for post-processing
            workspace_paths.append(str(workspace_path))

        except Exception as e:
            logger.error(f"Task {task_name} failed with error: {e}", exc_info=True)
            # Still add workspace path even if task failed (for post-processing to record failure)
            if 'workspace_path' in locals():
                workspace_paths.append(str(workspace_path))
            continue

    # Run post-processing to generate report
    logger.info("=" * 80)
    logger.info("Running Post-Processing")
    logger.info("=" * 80)

    try:
        post_processing_handler = load_post_processing_handler(agent, logger)
        post_processing_handler(workspace_paths, logger)
    except NotImplementedError as e:
        logger.warning(f"Post-processing skipped: {e}")
    except Exception as e:
        logger.error(f"Post-processing failed: {e}", exc_info=True)

    logger.info("=" * 80)
    logger.info("AgentKernelArena Framework Completed")
    logger.info("=" * 80)



if __name__ == "__main__":
    main()
