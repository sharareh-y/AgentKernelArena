# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import os
from typing import Any
import logging
from pathlib import Path
from datetime import datetime
import ast
import yaml
import json
import threading
import subprocess

from agents import register_agent
from agents.single_llm_call.llm_service import LLMService


def run_cmd_collect_output(
    cmd: str,
    workspace: str,
    logger: logging.Logger,
    timeout_seconds: int = 600
) -> str:
    """
    Encapsulate the original logic: run a command, stream outputs in real-time,
    parse JSON events, and finally return the combined stdout + stderr as a single string.
    """

    logger.info(f"Running command: {cmd}")
    logger.info("=" * 80)
    logger.info("Agent Output (streaming):")
    logger.info("=" * 80)

    process = subprocess.Popen(
        cmd,
        shell=True,  # nosec B602 -- shell=True is required to launch agent process
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace,
        bufsize=1
    )

    # Close stdin immediately to prevent the agent from waiting for user input
    if process.stdin:
        process.stdin.close()

    stdout_lines = []
    stderr_lines = []

    # ------------------------
    # Logic to parse agent events
    # ------------------------
    def format_agent_event(data):
        if not isinstance(data, dict):
            return str(data)

        event_type = data.get("type")

        if event_type == "assistant":
            content = data.get("message", {}).get("content", [])
            texts = [part.get("text", "") for part in content if isinstance(part, dict)]
            text = " ".join(t.strip() for t in texts if t.strip())
            return f"assistant: {text}"

        if event_type == "thinking":
            text = " ".join((data.get("text") or "").split())
            subtype = data.get("subtype")
            return f"thinking[{subtype}] {text}" if subtype else f"thinking {text}"

        if event_type == "tool_call":
            call = data.get("tool_call", {})
            name = next(iter(call.keys()), "unknown_tool")
            args = call.get(name, {}).get("args", {})
            return f"tool_call {name} {args}"

        # fallback
        return json.dumps(data, ensure_ascii=False)

    # ------------------------
    # Threaded streaming of stdout/stderr
    # ------------------------
    def read_stream(stream, output_list, prefix, log_func):
        try:
            for line in iter(stream.readline, ''):
                if not line:
                    break
                raw = line.rstrip()

                # Try to parse as JSON event
                try:
                    obj = json.loads(raw)
                    formatted = format_agent_event(obj)
                    if formatted:
                        output_list.append(formatted)
                        log_func(f"{prefix} {formatted}")
                        continue
                except Exception:
                    pass

                # Fallback: parse using literal_eval
                try:
                    obj = ast.literal_eval(raw)
                    formatted = format_agent_event(obj)
                    if formatted:
                        output_list.append(formatted)
                        log_func(f"{prefix} {formatted}")
                        continue
                except Exception:
                    pass

                # Plain text line
                if raw.strip():
                    output_list.append(raw)
                    log_func(f"{prefix} {raw}")

        finally:
            stream.close()

    # Start threads to read stdout and stderr concurrently
    t1 = threading.Thread(
        target=read_stream,
        args=(process.stdout, stdout_lines, "[AGENT]", logger.info),
        daemon=True
    )
    t2 = threading.Thread(
        target=read_stream,
        args=(process.stderr, stderr_lines, "[AGENT STDERR]", logger.warning),
        daemon=True
    )

    t1.start()
    t2.start()

    # Wait for the process to finish with timeout handling
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout {timeout_seconds}s. Terminating...")
        process.terminate()

    t1.join(timeout=1)
    t2.join(timeout=1)

    logger.info(f"Exit code: {process.returncode}")

    # Return the combined output
    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines)

    return output



@register_agent("single_llm_call")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str, provider: str = "claude", config_path: str = "agents/single_llm_call/agent_config.yaml") -> str:
    """
    Launch single LLM call agent.

    Args:
        eval_config: Evaluator settings passed from main (includes task metadata like task_type)
        task_config_dir: Path to the task configuration used to build the prompt
        workspace: Workspace directory (not used for single LLM call)
        provider: LLM provider (openai, claude, openrouter, vllm)
        config_path: Path to config.yaml

    Returns:
        str: LLM response content
    """
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Start to launch single LLM call evaluation:")
    logger.info("=" * 80)

    # setup common paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    workspace = os.path.abspath(os.path.join(project_root, workspace))

    # Create LLM service with config
    provider = eval_config['agent'].get('llm_provider', provider)
    llm_service = LLMService(config_path=config_path, provider=provider)

    # Make single LLM call
    # result = llm_service.simple_chat(prompt)

    # Read test config from task_config_dir/test_config.yaml
    test_config_path = Path(os.path.join(project_root, task_config_dir))
    with test_config_path.open("r") as f:
        test_config = yaml.safe_load(f) or {}

    compile_cmd = test_config.get("compile_command")
    exec_cmd = test_config.get("correctness_command")
    perf_cmd = test_config.get("performance_command")
    source_file_path = test_config.get("source_file_path")
    source_file_path = Path(workspace) / source_file_path
    target_file_path = test_config.get("target_file_path")
    target_file_path = Path(workspace) / target_file_path

    # Single llm call for task generation.
    instrution = test_config['prompt'].get("instructions", "")

    with source_file_path.open("r") as f:
        input_code = f.read() or ""

    prompt = instrution + '\n input pytroch code as follows:\n' + input_code
    gen_output = llm_service.simple_chat(prompt)
    print(f'[INFO] gen result:{gen_output}')
   
    with target_file_path.open("w") as f:
        f.write(gen_output)
    
    # Check the compile, correctness and speedup performance.
    agent_config_path = Path(os.path.join(project_root, config_path))
    with agent_config_path.open("r") as f:
        agent_config = yaml.safe_load(f) or {}
    timeout_seconds = int(agent_config.get("timeout_seconds", 600))

    results = ""
    results += run_cmd_collect_output(compile_cmd, workspace, logger)
    results += run_cmd_collect_output(exec_cmd, workspace, logger)
    results += run_cmd_collect_output(perf_cmd, workspace, logger)    

    # Post processing:Write task_result.yaml to workspace
    template_path = os.path.join(project_root, "src/prompts/task_result_template.yaml")
    with open(template_path, "r") as f:
        template_data = yaml.safe_load(f)

    # You may have a variable 'test_config' already loaded with info; if not, fallback to N/A
    task_name = test_config.get("source_file_path", "N/A") if "test_config" in locals() else "N/A"
    best_optimized_source_file_path = str(target_file_path)
    best_optimized_kernel_functions = test_config.get("target_kernel_functions", [])
    # Compose the result structure, overwriting/correcting relevant fields
    template_data["task_name"] = eval_config.get("tasks", "N/A")[0]
    template_data["best_optimized_source_file_path"] = best_optimized_source_file_path
    template_data["best_optimized_kernel_functions"] = best_optimized_kernel_functions
    template_data["task_type"] = eval_config.get('task_type') 

    # write performance data to task_result.yaml
    eval_result_path = Path(workspace) / "eval_result.yaml"
    with eval_result_path.open("r") as f:
        eval_results = yaml.safe_load(f)

    template_data["pass_compilation"] = eval_results.get('compiled', False)
    template_data["pass_correctness"] = eval_results.get('correctness', False)
    template_data["compilation_error_message"] = None
    template_data["correctness_error_message"] = None

    template_data["base_execution_time"] = eval_results.get('ori_time', 0.0)
    template_data["best_optimized_execution_time"] = eval_results.get('opt_time', 0.0)
    template_data["speedup_ratio"] = eval_results.get('speedup', 1.0)

    if template_data['speedup_ratio'] is not None and template_data["speedup_ratio"] < 1.0:
        template_data["speedup_ratio"] = 1.0
    # Write to workspace task_result.yaml
    # import pdb
    # pdb.set_trace()
    task_result_path = os.path.join(workspace, "task_result.yaml")
    with open(task_result_path, "w") as result_file:
        yaml.safe_dump(template_data, result_file, allow_unicode=True, sort_keys=False)


    return results
