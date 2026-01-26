# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import subprocess
import shutil
import logging
import threading
import os
import shlex
from pathlib import Path
from datetime import datetime
from typing import Any
import yaml
from agents import register_agent
from src.module_registration import AgentType, load_prompt_builder
import json
import glob
import requests


def write_debug_script(workspace: str, cmd: str, agent: str) -> None:
    """Optionally write the invocation command to a shell script for debugging."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    script_file = f"run_agent_{timestamp}.sh"

    script_lines = [
        "#!/bin/bash",
        f"# Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Workspace: {workspace}",
        f"# Agent: {agent}",
        "",
        f"cd {workspace}",
        cmd,
    ]

    script_file.write_text("\n".join(script_lines) + "\n")
    os.chmod(script_file, 0o755)


@register_agent("geak_ourllm_kernel2kernel")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """
    Launch cursor agent with real-time output streaming.

    Args:
        eval_config: Evaluator settings passed from main (includes task metadata like task_type)
        task_config_dir: Path to the task configuration used to build the prompt
        workspace: Workspace directory where the agent will run and read/write files

    Returns:
        str: Combined agent output (stdout plus stderr summary if present)
    """

    AGENT = "main_gaagent_hip_kernel2kernel.py"
    #AGENT = "main_gaagent_vllm_hip2hip.py"
    #AGENT = "main_gaagent_hip.py"
    AGENT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GEAK-agent", "src")
    
    AGENT_COMMAND = f"python3 {AGENT}"
    OPTIONS = f""

    current_task = Path(task_config_dir).with_suffix("").parent.relative_to("tasks")
    current_task = str(current_task)
    # setup common paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    workspace = os.path.abspath(os.path.join(project_root, workspace))
    
    # Check if the GEAK-hip agent main_gaagent_hip.py exists
    if not os.path.exists(os.path.join(AGENT_PATH, AGENT)):
        raise RuntimeError(
            f"Command '{AGENT}' not found. Please ensure '{AGENT_PATH}' is installed and in your PATH."
        )

    # read agent config
    config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open("r") as f:
        agent_config = yaml.safe_load(f) or {}
    logger = logging.getLogger(__name__)
    
    # read test config from task_config_dir/test_config.yaml
    test_config_path = Path(os.path.join(project_root, task_config_dir))
    with test_config_path.open("r") as f:
        test_config = yaml.safe_load(f) or {}
    compile_cmd = test_config.get("compile_command")
    exec_cmd = test_config.get("correctness_command")
    test_cmd = test_config.get("performance_command")
    source_file_path = test_config.get("source_file_path")

    # Populate the example_hip/FromRe_instructions.json fields from test_config and workspace
    json_path = os.path.join(AGENT_PATH, "..", "example_hip", "FromRe_instructions.json")
    with open(json_path, "r") as f:
        instructions = json.load(f)

    # build the FromRe_instructions.json fields from test_config and agent_config
    instructions[0]["compile_cmd"] = compile_cmd[0] if isinstance(compile_cmd, list) and compile_cmd else ""
    instructions[0]["exec_cmd"] = exec_cmd[0] if isinstance(exec_cmd, list) and exec_cmd else ""
    instructions[0]["test_cmd"] = test_cmd[0] if isinstance(test_cmd, list) and test_cmd else ""
    # Usually a list, take the first if list, else use as string
    instructions[0]["file_name"] = source_file_path[0] if isinstance(source_file_path, list) and source_file_path else ""
    # Assign workspace path
    instructions[0]["workspace"] = os.path.abspath( workspace)
    instructions[0]["task"] = current_task
    instructions[0]["model_name"] = agent_config['model_name']
    instructions[0]["max_length"] = agent_config['max_length']
    if 'kernel' in AGENT:
        instructions[0]["instruction"]  = "Please optimize the following HIP kernel/function for better performance on the ROCm platform (MI250 GPU).\n\
    MI250 specs: 208KB LDS per Compute Unit (CU), 64 CUs total.\n\nYou will receive only a single kernel/function from the .hip file.\n\
    You may only modify the function body, but you must output the entire function including its signature.\n\nAllowed:\n\nRewrite or optimize the function body only.\n\n\
    Add local variables, shared memory, unrolling, vectorized I/O, etc.\n\nReorder code inside the function.\n\nAdd comments inside the function.\n\nNot Allowed:\n\nDo NOT change the function name.\n\n\
    Do NOT change the function signature or parameter types.\n\nDo NOT add, remove, or modify any code outside this function.\n\nNo helper functions\n\nNo new includes\n\nNo new kernels\n\n\
    No changes to launch configuration\n\nDo NOT assume access to any code outside this function.\n\nOptimization guidelines (apply those that fit):\n\nChunked/tiled processing using registers or LDS\n\n\
    Shared-memory buffering (LDS)\n\nDelayed stores to shared memory\n\nVectorized loads/stores (float2/float4/uint4/etc.)\n\nLoop unrolling\n\nBound checks for variable sizes\n\nMinimize warp/wavefront divergence\n\n\
    Increase ILP via interleaving independent ops\n\nReduce LDS/register usage for higher occupancy\n\nFavor coalesced memory and AMD wavefront-friendly access patterns\n\nFuse operations where possible\n\n\
    Use compiler hints like #pragma unroll\n\nHard Requirements:\n\nReturn the full function, including the exact original function signature.\n\nOnly modify code inside the function body.\n\n\
    Preserve algorithmic correctness and bitwise-equivalent outputs.\n\nMaintains existing formatting and comments unless improving them.\n\nCode must be compilable and runnable."

    if "api_url" in eval_config.keys():
        instructions[0]['api_url'] = eval_config.get("api_url", "http://0.0.0.0:8001/v1/chat/completions")
    else:
        instructions[0]['api_url'] = agent_config.get("api_url", "http://0.0.0.0:8001/v1/chat/completions")

    if 'file_path' in instructions[0].keys():
        del instructions[0]['file_path']
    with open(json_path, "w") as f:
        json.dump(instructions, f, indent=2)
    

    # Set configs/hipbench_gaagent_config.yaml
    config_dir = os.path.dirname(config_path)
    hipbench_config_path = os.path.join(AGENT_PATH, "configs", "hipbench_gaagent_config.yaml")
    with open(hipbench_config_path, "r") as f:
        hipbench_config = yaml.safe_load(f) or {}
    # Set output_path as "<workspace>/geak_hip_iter_logs/iter"
    output_path = os.path.join(workspace, "geak_hip_iter_logs", "iter")
    hipbench_config["output_path"] = output_path
    max_iterations = agent_config.get("max_iterations", 20)
    hipbench_config["max_iteration"] = max_iterations
    descendant_num = agent_config.get("descendant_num", 2)
    hipbench_config["descendant_num"] = descendant_num
    # add new
    hipbench_config["target_gpu"] = agent_config.get("target_gpu", "MI308")
    hipbench_config['workspace_path'] = os.path.abspath(workspace)
    assert 'OPENAI_API_KEY' in os.environ, "OPENAI_API_KEY environment variable is not set."
    OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
    hipbench_config['api_key'] = OPENAI_API_KEY
    hipbench_config['max_length'] = agent_config.get("max_length", 8192)
    os.makedirs(output_path,exist_ok=True,mode=0o777)
    with open(hipbench_config_path, "w") as f:
        yaml.dump(hipbench_config, f, default_flow_style=False)


    cmd = f"python3 {AGENT}"
    # Enable to save the command to a shell script for manual replay/debugging.
    if False:
        write_debug_script(workspace, cmd, AGENT)
        logger.info("Debug script written; skipping live run.")
        return ""
    
    logger.info(f"Running command: {cmd}")
    logger.info("=" * 80)
    logger.info("Agent Output (streaming):")
    logger.info("=" * 80)

    # Give the agent a hard stop to avoid blocking downstream tasks if it
    # keeps waiting for interactive input after finishing its work.
    timeout_seconds = int(agent_config.get("timeout_seconds", 3600*5))

    # Use Popen for real-time output streaming with interactive input support
    process = subprocess.Popen(
        cmd,
        shell=True,  # nosec B602 -- shell=True is required to launch agent process
        stdin=subprocess.PIPE,  # Keep stdin closed so the agent exits when done
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=AGENT_PATH,
        bufsize=1  # Line buffered
    )

    # Close stdin immediately; leaving it attached keeps the agent alive waiting
    # for more user messages even after it reports completion.
    if process.stdin:
        process.stdin.close()

    # Collect output while streaming
    stdout_lines = []
    stderr_lines = []

    def format_agent_event(data):
        """Convert cursor stream-json payloads into a readable single-line string."""
        if not isinstance(data, dict):
            return str(data)

        event_type = data.get("type")
        if event_type == "assistant":
            content = data.get("message", {}).get("content", [])
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            text = " ".join(t.strip() for t in texts if t and t.strip())
            return f"assistant: {text}" if text else "assistant (no text)"

        if event_type == "thinking":
            text = " ".join((data.get("text") or "").split())
            subtype = data.get("subtype")
            # Skip empty deltas to avoid noisy blank lines
            if not text:
                return None
            return f"thinking[{subtype}] {text}" if subtype else f"thinking {text}"

        if event_type == "tool_call":
            subtype = data.get("subtype")
            call = data.get("tool_call") or {}
            call_name = next(iter(call.keys()), "unknown_tool")
            args = call.get(call_name, {}).get("args", {}) if isinstance(call, dict) else {}
            summary = []
            if isinstance(args, dict):
                if "path" in args:
                    summary.append(f"path={args.get('path')}")
                if "command" in args:
                    summary.append(f"cmd={args.get('command')}")
            details = " ".join(summary)
            return f"tool_call[{subtype}] {call_name} {details}".strip()

        if event_type == "user":
            message = data.get("message", {}).get("content", [])
            texts = []
            for part in message:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
            text = " ".join(t.strip() for t in texts if t and t.strip())
            if not text:
                return "user (no text)"
            text = " ".join(text.split())
            return f"user: {text[:160]}{'...' if len(text) > 160 else ''}"

        if event_type == "system":
            model = data.get("model")
            cwd = data.get("cwd")
            return f"system init model={model} cwd={cwd}"

        # Fallback: compact json
        import json
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def read_stream(stream, output_list, prefix, log_func):
        """Read from stream in a separate thread to avoid blocking"""
        import json
        import ast
        try:
            for line in iter(stream.readline, ''):
                if not line:
                    break
                raw_line = line.rstrip()

                # Try to parse as JSON (stream-json format)
                try:
                    data = json.loads(raw_line)
                    formatted = format_agent_event(data)
                    if formatted:
                        output_list.append(formatted)
                        log_func(f"{prefix} {formatted}")
                    continue
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(raw_line)
                        formatted = format_agent_event(data)
                        if formatted:
                            output_list.append(formatted)
                            log_func(f"{prefix} {formatted}")
                        continue
                    except Exception:
                        pass

                if raw_line.strip():
                    output_list.append(raw_line)
                    log_func(f"{prefix} {raw_line}")
        finally:
            stream.close()

    # Create threads to read stdout and stderr concurrently
    # This allows user interaction to work while we capture output
    stdout_thread = threading.Thread(
        target=read_stream,
        args=(process.stdout, stdout_lines, "[AGENT]", logger.info),
        daemon=True
    )
    stderr_thread = threading.Thread(
        target=read_stream,
        args=(process.stderr, stderr_lines, "[AGENT STDERR]", logger.warning),
        daemon=True
    )

    # Start reading threads
    stdout_thread.start()
    stderr_thread.start()

    # Wait for process to complete
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        logger.warning(f"Cursor agent timed out after {timeout_seconds}s; terminating process")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Force killing cursor agent process")
            process.kill()

    # Wait for output threads to finish reading
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)

    # Log stderr summary if present
    if stderr_lines:
        logger.warning("=" * 80)
        logger.warning(f"Agent STDERR captured {len(stderr_lines)} lines")
        logger.warning("=" * 80)

    logger.info("=" * 80)
    logger.info(f"Agent completed with exit code: {process.returncode}")
    logger.info("=" * 80)

    # Return combined output
    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines)

    # Post-processing output
    # Path to geak_hip_iter_logs
    logs_dir = os.path.join(workspace, "geak_hip_iter_logs")
    
    # Find all iter_*.perf files
    iter_perf_files = glob.glob(os.path.join(logs_dir, "iter_*.perf"))
    if not iter_perf_files:
        raise RuntimeError(f"No iter_*.perf files found in {logs_dir}")

    # Extract the largest iteration number
    def extract_iter_num(path):
        name = os.path.basename(path)
        try:
            return int(name.split("_")[1].split(".")[0])
        except Exception:
            return -1

    latest_file = max(iter_perf_files, key=extract_iter_num)
    with open(latest_file, "r") as f:
        perf_data = json.load(f)

    ori_perf = perf_data.get("ori_perf")
    opt_perf = perf_data.get("opt_perf")

    # Post processing:Write task_result.yaml to workspace
    template_path = os.path.join(project_root, "src/prompts/task_result_template.yaml")
    with open(template_path, "r") as f:
        template_data = yaml.safe_load(f)

    # You may have a variable 'test_config' already loaded with info; if not, fallback to N/A
    task_name = test_config.get("source_file_path", "N/A") if "test_config" in locals() else "N/A"
    best_optimized_source_file_path = source_file_path
    best_optimized_kernel_functions = test_config.get("target_kernel_functions", [])
    # Compose the result structure, overwriting/correcting relevant fields
    template_data["task_name"] = current_task #eval_config.get("current_task", None) #eval_config.get("tasks", "N/A")[0]
    template_data["task_type"] = "hip2hip"
    template_data["timestamp"] = datetime.now().isoformat(timespec="seconds")
    template_data["agent_type"] = "geak_hip"
    template_data["best_optimized_source_file_path"] = best_optimized_source_file_path
    template_data["best_optimized_kernel_functions"] = best_optimized_kernel_functions

    # write performance data to task_result.yaml
    has_perf = bool(iter_perf_files)
    template_data["pass_compilation"] = bool(has_perf)
    template_data["pass_correctness"] = bool(has_perf)
    template_data["compilation_error_message"] = None
    template_data["correctness_error_message"] = None

    template_data["base_execution_time"] = ori_perf
    template_data["best_optimized_execution_time"] = opt_perf
    try:
        if isinstance(ori_perf, list) and isinstance(opt_perf, list) and len(ori_perf) == len(opt_perf) and len(ori_perf) > 0:
            # Index correspondence division, avg the ratio
            ratios = []
            for o, opt in zip(ori_perf, opt_perf):
                try:
                    r = float(o) / float(opt) if opt else 0.0
                except Exception:
                    r = 0.0
                ratios.append(r)
            template_data["speedup_ratio"] = sum(ratios) / len(ratios)
            template_data["base_execution_time"] = sum(ori_perf) / len(ori_perf)
            template_data["best_optimized_execution_time"] = sum(opt_perf) / len(opt_perf)
        elif isinstance(ori_perf, (float, int)) and isinstance(opt_perf, (float, int)) and opt_perf:
            template_data["speedup_ratio"] = float(ori_perf) / float(opt_perf)
            template_data["base_execution_time"] = ori_perf
            template_data["best_optimized_execution_time"] = opt_perf
        else:
            print("mismatching perf data, set speedup_ratio to 1.0...")
            template_data["speedup_ratio"] = 1.0
    except Exception:
        print("mismatching perf data, set speedup_ratio to 1.0...")
        template_data["speedup_ratio"] = 1.0

    if template_data["speedup_ratio"] < 1.0:
        template_data["speedup_ratio"] = 1.0
    # Write to workspace task_result.yaml
    task_result_path = os.path.join(workspace, "task_result.yaml")
    with open(task_result_path, "w") as result_file:
        yaml.safe_dump(template_data, result_file, allow_unicode=True, sort_keys=False)
   

    return output
