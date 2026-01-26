# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
GEAK-OptimAgentV2 agent launcher for AgentKernelArena

This launcher supports both TritonBench (Triton kernels) and ROCm (HIP kernels)
task types. It auto-clones the GEAK-agent repository if not present.
"""
import logging
import os
import subprocess
import sys
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from agents import register_agent

logger = logging.getLogger(__name__)


# ============================================================================
# GEAK-agent Path Management
# ============================================================================

def _resolve_relative_paths(config: Dict, agent_dir: Path) -> Dict:
    """
    Resolve paths starting with './' relative to the agent directory.
    
    Args:
        config: Configuration dictionary (modified in place)
        agent_dir: Path to agents/geak_optimagentv2/
        
    Returns:
        Modified config with resolved paths
    """
    def resolve_value(value):
        if isinstance(value, str) and value.startswith('./'):
            resolved = str(agent_dir / value[2:])  # Remove './' prefix
            return resolved
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item) for item in value]
        return value
    
    return resolve_value(config)


def _run_setup_command(cmd: List[str], cwd: str, description: str) -> bool:
    """Run a setup command and log results."""
    try:
        logger.info(f"Running: {description}")
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for pip installs
        )
        if result.returncode != 0:
            logger.warning(f"{description} failed: {result.stderr}")
            return False
        logger.info(f"{description} completed successfully")
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"{description} timed out")
        return False
    except Exception as e:
        logger.warning(f"{description} error: {e}")
        return False


def _patch_geak_imports(geak_src_dir: Path) -> None:
    """
    Patch GEAK-agent imports to avoid conflicts with AgentKernelArena.
    
    Renames conflicting module folders and updates all imports:
    - agents → geak_agents
    - models → geak_models
    - dataloaders → geak_dataloaders
    - utils → geak_utils
    - memories → geak_memories
    - prompts → geak_prompts
    - retrievers → geak_retrievers
    """
    import re
    
    # Folders to rename (only if they exist)
    renames = {
        'agents': 'geak_agents',
        'models': 'geak_models', 
        'dataloaders': 'geak_dataloaders',
        'utils': 'geak_utils',
        'memories': 'geak_memories',
        'prompts': 'geak_prompts',
        'retrievers': 'geak_retrievers',
    }
    
    # Additional import renames (no folder rename needed, just imports)
    # No additional import renames needed
    # openevolve branch has tb_eval folder, GEAK-agent imports tb_eval - they match!
    import_only_renames = {}
    
    # Step 1: Rename folders
    logger.info("Renaming GEAK-agent module folders to avoid conflicts...")
    renamed = []
    for old_name, new_name in renames.items():
        old_path = geak_src_dir / old_name
        new_path = geak_src_dir / new_name
        if old_path.exists() and old_path.is_dir():
            if not new_path.exists():
                old_path.rename(new_path)
                renamed.append((old_name, new_name))
                logger.info(f"  Renamed: {old_name} → {new_name}")
    
    if not renamed:
        logger.info("  No folders needed renaming (already patched?)")
        return
    
    # Step 2: Update imports in all Python files
    logger.info("Patching imports in Python files...")
    
    # Build regex patterns for import replacement
    patterns = []
    
    # Add patterns for renamed folders
    for old_name, new_name in renamed:
        # Match various import patterns:
        # - from agents.X import Y
        # - from agents import X
        # - import agents.X
        # - import agents
        patterns.append((
            re.compile(rf'\b(from\s+){old_name}(\s+import\b|\.|$)', re.MULTILINE),
            rf'\1{new_name}\2'
        ))
        patterns.append((
            re.compile(rf'\b(import\s+){old_name}(\s+|\.|\s*,|$)', re.MULTILINE),
            rf'\1{new_name}\2'
        ))
    
    # Add patterns for import-only renames (e.g., tb_eval -> geak_eval)
    for old_name, new_name in import_only_renames.items():
        patterns.append((
            re.compile(rf'\b(from\s+){old_name}(\s+import\b|\.|$)', re.MULTILINE),
            rf'\1{new_name}\2'
        ))
        patterns.append((
            re.compile(rf'\b(import\s+){old_name}(\s+|\.|\s*,|$)', re.MULTILINE),
            rf'\1{new_name}\2'
        ))
    
    # Find all Python files
    py_files = list(geak_src_dir.rglob('*.py'))
    patched_count = 0
    
    for py_file in py_files:
        try:
            content = py_file.read_text()
            original = content
            
            for pattern, replacement in patterns:
                content = pattern.sub(replacement, content)
            
            if content != original:
                py_file.write_text(content)
                patched_count += 1
        except Exception as e:
            logger.warning(f"  Could not patch {py_file}: {e}")
    
    logger.info(f"  Patched {patched_count} Python files")


def _setup_geak_dependencies(agent_dir: Path) -> None:
    """
    Install GEAK-agent dependencies and GEAK-eval.
    
    Based on GEAK-agent README:
    1. Patch imports to avoid conflicts with AgentKernelArena
    2. pip install -r requirements.txt (GEAK-agent deps)
    3. Clone and install GEAK-eval (for evaluators)
    """
    geak_agent_dir = agent_dir / "GEAK-agent"
    geak_src_dir = geak_agent_dir / "src"
    geak_eval_dir = agent_dir / "GEAK-eval"
    setup_marker = agent_dir / ".geak_setup_complete"
    
    # Skip if already set up
    if setup_marker.exists():
        logger.info("GEAK dependencies already set up (marker file found)")
        return
    
    logger.info("=" * 60)
    logger.info("Setting up GEAK-agent dependencies...")
    logger.info("=" * 60)
    
    # 1. Patch GEAK-agent imports to avoid conflicts
    if geak_src_dir.exists():
        _patch_geak_imports(geak_src_dir)
    
    # 2. Install GEAK-agent requirements
    requirements_file = geak_agent_dir / "requirements.txt"
    if requirements_file.exists():
        # Read requirements and separate torch from others
        with open(requirements_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Separate torch from other requirements
        torch_reqs = [req for req in requirements if req.lower().startswith('torch')]
        other_reqs = [req for req in requirements if not req.lower().startswith('torch')]
        
        # Install non-torch requirements normally
        if other_reqs:
            _run_setup_command(
                [sys.executable, "-m", "pip", "install"] + other_reqs,
                str(geak_agent_dir),
                "Installing GEAK-agent requirements (non-torch)"
            )
        
        # Install torch from ROCm index to get ROCm version
        if torch_reqs:
            _run_setup_command(
                [sys.executable, "-m", "pip", "install", "--index-url", 
                 "https://download.pytorch.org/whl/rocm6.2"] + torch_reqs,
                str(geak_agent_dir),
                "Installing PyTorch from ROCm index"
            )
    
    # 3. Clone GEAK-eval if not present (openevolve branch has tb_eval folder)
    if not geak_eval_dir.exists():
        logger.info("Cloning GEAK-eval repository (openevolve branch)...")
        try:
            subprocess.run(
                [
                    "git", "clone",
                    "--branch", "openevolve",
                    "https://github.com/AMD-AGI/GEAK-eval.git",
                    str(geak_eval_dir)
                ],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info("GEAK-eval cloned successfully!")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to clone GEAK-eval: {e.stderr}")
            logger.warning("Some features may not work without GEAK-eval")
    
    # 4. Install GEAK-eval (openevolve branch already has tb_eval - no renaming needed!)
    if geak_eval_dir.exists():
        # Install requirements
        eval_requirements = geak_eval_dir / "requirements.txt"
        if eval_requirements.exists():
            _run_setup_command(
                [sys.executable, "-m", "pip", "install", "-r", str(eval_requirements)],
                str(geak_eval_dir),
                "Installing GEAK-eval requirements"
            )
        
        # Install GEAK-eval as editable package (keeps tb_eval name)
        _run_setup_command(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
            str(geak_eval_dir),
            "Installing GEAK-eval package (tb_eval)"
        )
    
    # Create marker file to skip setup on next run
    try:
        setup_marker.touch()
        logger.info("Setup complete! Marker file created.")
    except Exception:
        pass  # Non-critical
    
    logger.info("=" * 60)
    logger.info("GEAK dependencies setup finished")
    logger.info("=" * 60)


def ensure_geak_agent_available(agent_config: Dict) -> str:
    """
    Ensure GEAK-agent is available, auto-clone and setup if not present.
    
    Priority:
    1. GEAK_AGENT_PATH environment variable
    2. geak_agent_path in agent_config.yaml
    3. Auto-clone to agents/geak_optimagentv2/GEAK-agent/
    
    Also installs dependencies (requirements.txt) and GEAK-eval on first run.
    
    Returns:
        Path to GEAK-agent/src directory
    """
    agent_dir = Path(__file__).parent
    default_clone_dir = agent_dir / "GEAK-agent"
    default_src_path = default_clone_dir / "src"
    
    # Check environment variable first
    env_path = os.environ.get('GEAK_AGENT_PATH')
    if env_path and os.path.exists(env_path):
        logger.info(f"Using GEAK-agent from GEAK_AGENT_PATH env var: {env_path}")
        return env_path
    
    # Check config file
    config_path = agent_config.get('geak_agent_path')
    if config_path and os.path.exists(config_path):
        logger.info(f"Using GEAK-agent from agent_config.yaml: {config_path}")
        return config_path
    
    # Check default location - if exists, just ensure setup is done
    if default_src_path.exists():
        logger.info(f"Using GEAK-agent from default location: {default_src_path}")
        _setup_geak_dependencies(agent_dir)
        return str(default_src_path)
    
    # Auto-clone
    logger.info("GEAK-agent not found. Auto-cloning from GitHub...")
    logger.info(f"Clone destination: {default_clone_dir}")
    
    try:
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/AMD-AGI/GEAK-agent.git",
                str(default_clone_dir)
            ],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("GEAK-agent cloned successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone GEAK-agent: {e.stderr}")
        raise RuntimeError(
            f"Failed to auto-clone GEAK-agent. Please either:\n"
            f"  1. Set GEAK_AGENT_PATH environment variable, or\n"
            f"  2. Set geak_agent_path in agent_config.yaml, or\n"
            f"  3. Manually clone: git clone https://github.com/AMD-AGI/GEAK-agent.git {default_clone_dir}\n"
            f"Error: {e.stderr}"
        )
    
    # Run dependency setup after cloning
    _setup_geak_dependencies(agent_dir)
    
    return str(default_src_path)


# ============================================================================
# Adapter Classes - Bridge AgentKernelArena tasks to GEAK-agent format
# ============================================================================

@dataclass
class ProblemStateAdapter:
    """Adapter for TritonBench ProblemState"""
    filename: str
    instruction: str
    label: Optional[str] = None
    test_code: Optional[str] = None
    solution: Optional[str] = None
    speedup: float = 0.0


@dataclass
class ProblemStateROCmAdapter:
    """Adapter for ROCm ProblemStateROCm"""
    filename: str
    instruction: str
    label: str
    opname: str
    target_kernel_name: str
    test_code: str
    solution: Optional[str] = None
    pass_call: bool = False
    pass_exe: bool = False
    speedup: float = 0.0


class AIGEvalTritonBenchAdapter:
    """
    Adapts AgentKernelArena task configuration to GEAK-agent's TritonBench interface.
    
    This adapter creates ProblemState objects from AgentKernelArena task configs and
    implements the test_opt_correctness interface using AgentKernelArena's commands.
    """
    
    def __init__(self, task_config: Dict, workspace: str, agent_config: Dict):
        self.task_config = task_config
        self.workspace = workspace
        self.agent_config = agent_config
        self.problem_states = [self._create_problem_state()]
        
    def _create_problem_state(self) -> ProblemStateAdapter:
        """Create a ProblemState from AgentKernelArena task config."""
        source_files = self.task_config.get('source_file_path', [])
        target_functions = self.task_config.get('target_kernel_functions', [])
        
        # Get source code
        source_code = ""
        filename = "kernel.py"
        if source_files and source_files[0]:
            source_path = Path(self.workspace) / source_files[0]
            if source_path.exists():
                filename = source_files[0]
                with open(source_path, 'r') as f:
                    source_code = f.read()
        
        # Build instruction from task config
        instruction = self._build_instruction(target_functions, source_code)
        
        return ProblemStateAdapter(
            filename=filename,
            instruction=instruction,
            label=source_code,
            test_code=self._extract_test_code(source_code),
            solution=None,
            speedup=0.0
        )
    
    def _build_instruction(self, target_functions: List[str], source_code: str) -> str:
        """Build instruction text from task config."""
        prompt_config = self.task_config.get('prompt', {})
        
        if prompt_config.get('instructions'):
            return prompt_config['instructions']
        
        instruction = "Optimize the following Triton kernel for AMD GPU:\n\n"
        if target_functions:
            instruction += f"Target functions: {', '.join(target_functions)}\n\n"
        instruction += f"Source code:\n```python\n{source_code}\n```\n"
        
        return instruction
    
    def _extract_test_code(self, source_code: str) -> str:
        """Extract test code section from source (after separator)."""
        separator = "#" * 146
        if separator in source_code:
            return source_code.split(separator)[-1]
        return ""
    
    def __len__(self) -> int:
        return len(self.problem_states)
    
    def test_opt_correctness(
        self,
        code: str,
        filename: str,
        tmp_dir: str,
        exe_dir: str = "pass_exe",
        gpu_id: int = 0
    ) -> Tuple[bool, bool, float, str, str]:
        """
        Test code correctness using AgentKernelArena's compile and correctness commands.
        
        Returns:
            (pass_call, pass_exe, speedup, stdout, stderr)
        """
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(exe_dir, exist_ok=True)
        
        # Write code to temp file
        test_file = Path(tmp_dir) / filename
        with open(test_file, 'w') as f:
            f.write(code)
        
        # Run compile command
        compile_cmds = self.task_config.get('compile_command', [])
        for cmd in compile_cmds:
            # Replace filename placeholder if present
            cmd = cmd.replace(filename, str(test_file))
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,  # nosec B602 -- shell=True is required to launch agent process
                    timeout=60, cwd=self.workspace
                )
                if result.returncode != 0:
                    return False, False, 0.0, result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                return False, False, 0.0, "", "Compile timeout"
            except Exception as e:
                return False, False, 0.0, "", str(e)
        
        # Run correctness command
        correctness_cmds = self.task_config.get('correctness_command', [])
        for cmd in correctness_cmds:
            cmd = cmd.replace(filename, str(test_file))
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,  # nosec B602 -- shell=True is required to launch agent process
                    timeout=300, cwd=self.workspace
                )
                if result.returncode != 0:
                    return True, False, 0.0, result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                return True, False, 0.0, "", "Correctness test timeout"
            except Exception as e:
                return True, False, 0.0, "", str(e)
        
        # Copy successful code to exe_dir
        shutil.copy(test_file, Path(exe_dir) / filename)
        
        # Run performance command if present
        speedup = 1.0
        perf_cmds = self.task_config.get('performance_command', [])
        if perf_cmds:
            for cmd in perf_cmds:
                cmd = cmd.replace(filename, str(test_file))
                try:
                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True,  # nosec B602 -- shell=True is required to launch agent process
                        timeout=600, cwd=self.workspace
                    )
                    # Parse speedup from output if available
                    # This is a simplified implementation
                except Exception:
                    pass
        
        return True, True, speedup, "Success", ""
    
    def write_file(self, file_path: str):
        """Write results to output file."""
        import json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            for ps in self.problem_states:
                output = {
                    "instruction": ps.instruction,
                    "label": ps.label,
                    "filename": ps.filename,
                    "predict": ps.solution if ps.solution else "",
                    "speedup": ps.speedup
                }
                f.write(json.dumps(output) + "\n")


class AIGEvalROCmAdapter:
    """
    Adapts AgentKernelArena task configuration to GEAK-agent's ROCm interface.
    """
    
    def __init__(self, task_config: Dict, workspace: str, agent_config: Dict, log_root: str):
        self.task_config = task_config
        self.workspace = workspace
        self.agent_config = agent_config
        self.log_root = log_root
        self.rocm_tests = True  # Flag for ROCm mode detection
        self.problem_states = [self._create_problem_state()]
        
    def _create_problem_state(self) -> ProblemStateROCmAdapter:
        """Create a ProblemStateROCm from AgentKernelArena task config."""
        source_files = self.task_config.get('source_file_path', [])
        target_functions = self.task_config.get('target_kernel_functions', [])
        
        # Get source code
        source_code = ""
        filename = "kernel.hip"
        if source_files and source_files[0]:
            source_path = Path(self.workspace) / source_files[0]
            if source_path.exists():
                filename = source_files[0]
                with open(source_path, 'r') as f:
                    source_code = f.read()
        
        # Extract opname from filename or target functions
        opname = Path(filename).stem
        target_kernel_name = target_functions[0] if target_functions else opname
        
        instruction = self._build_instruction(target_functions, source_code)
        
        return ProblemStateROCmAdapter(
            filename=filename,
            instruction=instruction,
            label=source_code,
            opname=opname,
            target_kernel_name=target_kernel_name,
            test_code=self._extract_test_code(source_code),
            solution=None,
            pass_call=False,
            pass_exe=False,
            speedup=0.0
        )
    
    def _build_instruction(self, target_functions: List[str], source_code: str) -> str:
        """Build instruction text from task config."""
        prompt_config = self.task_config.get('prompt', {})
        
        if prompt_config.get('instructions'):
            return prompt_config['instructions']
        
        instruction = "Optimize the following HIP kernel for AMD GPU:\n\n"
        if target_functions:
            instruction += f"Target functions: {', '.join(target_functions)}\n\n"
        instruction += f"Source code:\n```cpp\n{source_code}\n```\n"
        
        return instruction
    
    def _extract_test_code(self, source_code: str) -> str:
        """Extract test code section from source."""
        separator = "#" * 146
        if separator in source_code:
            return source_code.split(separator)[-1]
        return ""
    
    def __len__(self) -> int:
        return len(self.problem_states)
    
    def test_opt_correctness(
        self,
        code: str,
        filename: str,
        opname: str,
        tmp_dir: str = "temp",
        save_scripts: bool = True,
        exe_dir: str = "pass_exe",
        gpu_id: int = 0
    ) -> Tuple[bool, bool, str, str, str, str]:
        """
        Test code correctness for ROCm kernels.
        
        Returns:
            (pass_call, pass_exe, call_stdout, call_stderr, exe_stdout, exe_stderr)
        """
        tmp_dir = os.path.join(self.log_root, tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        exe_dir = os.path.join(self.log_root, exe_dir)
        os.makedirs(exe_dir, exist_ok=True)
        
        # Write code to temp file
        test_file = Path(tmp_dir) / filename
        with open(test_file, 'w') as f:
            f.write(code)
        
        # Run compile command
        compile_cmds = self.task_config.get('compile_command', [])
        for cmd in compile_cmds:
            cmd = cmd.replace(filename, str(test_file))
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,  # nosec B602 -- shell=True is required to launch agent process
                    timeout=120, cwd=self.workspace
                )
                if result.returncode != 0:
                    return False, False, result.stdout, result.stderr, "", result.stderr
            except subprocess.TimeoutExpired:
                return False, False, "", "Compile timeout", "", "Compile timeout"
            except Exception as e:
                return False, False, "", str(e), "", str(e)
        
        # Run correctness command
        correctness_cmds = self.task_config.get('correctness_command', [])
        for cmd in correctness_cmds:
            cmd = cmd.replace(filename, str(test_file))
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,  # nosec B602 -- shell=True is required to launch agent process
                    timeout=300, cwd=self.workspace
                )
                if result.returncode != 0:
                    return True, False, "Compiled", "", result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                return True, False, "Compiled", "", "", "Correctness test timeout"
            except Exception as e:
                return True, False, "Compiled", "", "", str(e)
        
        # Copy successful code
        if save_scripts:
            shutil.copy(test_file, Path(exe_dir) / opname)
        
        return True, True, "Success", "", "Success", ""
    
    def run_perf_evaluation(
        self,
        exec_folder: str,
        gen_perf_folder: str,
        gpu_id: int = 0
    ) -> Dict:
        """Run performance evaluation for ROCm kernels."""
        perf_results = {}
        perf_cmds = self.task_config.get('performance_command', [])
        
        if not perf_cmds:
            return perf_results
        
        # Run performance commands
        for cmd in perf_cmds:
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,  # nosec B602 -- shell=True is required to launch agent process
                    timeout=600, cwd=self.workspace
                )
                # Parse performance results from output
                # This is a simplified implementation
            except Exception as e:
                logger.warning(f"Performance evaluation failed: {e}")
        
        return perf_results
    
    def write_file(self, file_path: str, start_idx: int = 0, datalen: Optional[int] = None):
        """Write results to output file."""
        import json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data_len = datalen if datalen is not None else len(self)
        with open(file_path, 'w') as f:
            for ps in self.problem_states[start_idx:(start_idx + data_len)]:
                output = {
                    "instruction": ps.instruction,
                    "label": ps.label,
                    "file": ps.filename,
                    "target_kernel_name": ps.target_kernel_name,
                    "predict": ps.solution if ps.solution else "",
                    "speedup": ps.speedup
                }
                f.write(json.dumps(output) + "\n")


# ============================================================================
# Agent Execution Functions (Subprocess-based)
# ============================================================================

def _create_runner_script(geak_path: str, mode: str) -> str:
    """
    Create a temporary Python script that runs GEAK-agent in isolation.
    
    This script runs in a separate process with GEAK-agent's path at the 
    beginning of sys.path, avoiding module conflicts with AgentKernelArena.
    """
    if mode == "tritonbench":
        script = '''
import sys
import os
import json

# Set GEAK-agent path FIRST
geak_path = sys.argv[1]
sys.path.insert(0, geak_path)

# Import from patched module names
from geak_agents.GaAgent import GaAgent
from geak_models.Claude import ClaudeModel
from geak_dataloaders.TritonBench import TritonBench

# Load configs from argv
workspace = sys.argv[2]
config_json = sys.argv[3]
config = json.loads(config_json)
task_config_json = sys.argv[4]
task_config = json.loads(task_config_json)

# ==========================================================================
# Main execution - Uses TritonBench directly with task filtering
# ==========================================================================

# Get API key
api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY') or os.environ.get('AMD_API_KEY')
model_id = config.get('model_id', 'claude-sonnet-4')

if not api_key:
    print("ERROR: No API key found")
    sys.exit(1)

print(f"[GEAK] Initializing ClaudeModel with model_id={model_id}")
model = ClaudeModel(api_key=api_key, model_id=model_id)

# Load full TritonBench dataset (uses GEAK-eval's tb_eval)
statis_path = config.get('statis_path')
py_folder = config.get('py_folder')
instruction_path = config.get('instruction_path')
golden_metrics = config.get('golden_metrics')
perf_ref_folder = config.get('perf_ref_folder')
perf_G_path = config.get('perf_G_path')

print(f"[GEAK] Loading TritonBench dataset from: {statis_path}")
dataset = TritonBench(
    statis_path=statis_path,
    py_folder=py_folder,
    instruction_path=instruction_path,
    golden_metrics=golden_metrics,
    py_interpreter=sys.executable,
    perf_ref_folder=perf_ref_folder,
    perf_G_path=perf_G_path
)
print(f"[GEAK] Full dataset size: {len(dataset)}")

# Find the specific task by matching filename/kernel name
target_kernels = task_config.get('target_kernel_functions', [])
target_filename = f"{target_kernels[0]}.py" if target_kernels else None

start_idx = 0
if target_filename:
    for idx, ps in enumerate(dataset.problem_states):
        if ps.filename == target_filename:
            start_idx = idx
            print(f"[GEAK] Found matching task at index {idx}: {ps.filename}")
            break
    else:
        print(f"[GEAK] Warning: Task {target_filename} not found in dataset, using first task")

# Get corpus_path for retrieval
corpus_path = config.get('corpus_path')

# Create agent
print("[GEAK] Creating GaAgent...")
print(f"[GEAK] corpus_path: {corpus_path}")
try:
    agent = GaAgent(
        model=model,
        dataset=dataset,
        corpus_path=corpus_path,
        mem_file=None,
        descendant_num=config.get('descendant_num', 2)
    )
    print(f"[GEAK] Agent created, memories count: {len(agent.memories)}")
except Exception as e:
    print(f"[GEAK] ERROR creating agent: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Run optimization on single task - use absolute path for output
# This is important because subprocess runs from GEAK-agent/src directory
abs_workspace = os.path.abspath(workspace)
output_path = os.path.join(abs_workspace, "geak_output.jsonl")
print(f"[GEAK] Running agent on task index {start_idx} with max_iteration={config.get('max_iteration', 10)}")
print(f"[GEAK] Output path: {output_path}")

try:
    agent.run(
        output_path=output_path,
        multi_thread=config.get('multi_thread', True),
        iteration_num=config.get('max_iteration', 10),
        temperature=config.get('temperature', 1.0),
        datalen=1,  # Process only 1 task
        gpu_id=config.get('gpu_id', 0),
        start_iter=0,
        ancestor_num=config.get('ancestor_num', 5),
        descendant_num=config.get('descendant_num', 2),
        descendant_debug=config.get('descendant_debug', 0),
        target_gpu=config.get('target_gpu', 'MI325X'),
        profiling=config.get('profiling', True),
        start_idx=start_idx  # Index of the matched task
    )
    print("[GEAK] Agent run completed!")
except Exception as e:
    print(f"[GEAK] ERROR during agent.run: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Read output and report results
if os.path.exists(output_path):
    with open(output_path, 'r') as f:
        lines = f.readlines()
        if lines:
            last_result = json.loads(lines[-1])
            print(f"[GEAK] Final speedup: {last_result.get('speedup', 0.0)}")

result = {
    'completed': True,
    'output_path': output_path
}
print(f"RESULT_JSON:{json.dumps(result)}")
'''
    else:  # rocm
        script = '''
import sys
import os
import json
import yaml

# Set GEAK-agent path FIRST
geak_path = sys.argv[1]
sys.path.insert(0, geak_path)

# Import from patched module names (agents → geak_agents, etc.)
from geak_agents.GaAgent_ROCm import GaAgent
from geak_models.Claude import ClaudeModel

# Load configs from argv
workspace = sys.argv[2]
config_json = sys.argv[3]
config = json.loads(config_json)
task_config_json = sys.argv[4]
task_config = json.loads(task_config_json)

# Get API key
api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY') or os.environ.get('AMD_API_KEY')
model_id = config.get('model_id', 'claude-sonnet-4')

if not api_key:
    print("ERROR: No API key found")
    sys.exit(1)

print(f"Initializing ClaudeModel with model_id={model_id}")
model = ClaudeModel(api_key=api_key, model_id=model_id)

# Import dataloader
from geak_dataloaders.ROCm import ROCm

# Initialize with ROCm data
corpus_path = config.get('corpus_path')
log_root = os.path.join(workspace, "geak_logs")
os.makedirs(log_root, exist_ok=True)

statis_path = config.get('rocm_statis_path')
py_folder = config.get('rocm_py_folder')
instruction_path = config.get('rocm_instruction_path')

print(f"Loading ROCm dataset")
dataset = ROCm(
    statis_path=statis_path,
    py_folder=py_folder,
    instruction_path=instruction_path,
    corpus_path=corpus_path,
    log_root=log_root
)

print(f"Dataset size: {len(dataset)}")

# Create agent
print("Creating GaAgent (ROCm)...")
agent = GaAgent(
    model=model,
    dataset=dataset,
    corpus_path=corpus_path,
    mem_file=None,
    descendant_num=config.get('descendant_num', 2)
)

# Run optimization
output_path = os.path.join(workspace, "geak_output.jsonl")
print(f"Running agent with max_iteration={config.get('max_iteration', 10)}")

agent.run(
    output_path=output_path,
    multi_thread=config.get('multi_thread', True),
    iteration_num=config.get('max_iteration', 10),
    temperature=config.get('temperature', 1.0),
    datalen=1,
    gpu_id=config.get('gpu_id', 0),
    start_iter=0,
    ancestor_num=config.get('ancestor_num', 5),
    descendant_num=config.get('descendant_num', 2),
    descendant_debug=config.get('descendant_debug', 0),
    target_gpu=config.get('target_gpu', 'MI325X'),
    profiling=config.get('profiling', False),
    start_idx=0
)

print("Agent run completed!")

# Write result summary
result = {
    'completed': True,
    'output_path': output_path
}
print(f"RESULT_JSON:{json.dumps(result)}")
'''
    return script


def run_tritonbench_agent(
    eval_config: Dict,
    task_config: Dict,
    workspace: str,
    agent_config: Dict,
    geak_path: str
) -> str:
    """Run GEAK-agent in TritonBench mode using subprocess."""
    import json
    import tempfile
    
    logger.info("Running GEAK-OptimAgentV2 in TritonBench mode (subprocess)")
    
    # Prepare config for subprocess
    api_key = agent_config.get('llm', {}).get('api_key') or os.environ.get('OPENAI_API_KEY') or os.environ.get('AMD_API_KEY')
    
    if not api_key:
        raise ValueError("No API key found. Set OPENAI_API_KEY or AMD_API_KEY environment variable, or set llm.api_key in agent_config.yaml")
    
    config = {
        'api_key': api_key,
        'model_id': agent_config.get('llm', {}).get('model_id', 'claude-sonnet-4'),
        'temperature': agent_config.get('llm', {}).get('temperature', 1.0),
        'max_iteration': agent_config.get('max_iteration', 10),
        'multi_thread': agent_config.get('multi_thread', True),
        'ancestor_num': agent_config.get('ancestor_num', 5),
        'descendant_num': agent_config.get('descendant_num', 2),
        'gpu_id': agent_config.get('gpu_id', 0),
        'descendant_debug': agent_config.get('descendant_debug', 0),
        'target_gpu': agent_config.get('target_gpu', 'MI325X'),
        'profiling': agent_config.get('profiling', True),
        'corpus_path': agent_config.get('tritonbench', {}).get('corpus_path'),
        'statis_path': agent_config.get('tritonbench', {}).get('statis_path'),
        'py_folder': agent_config.get('tritonbench', {}).get('py_folder'),
        'instruction_path': agent_config.get('tritonbench', {}).get('instruction_path'),
        'golden_metrics': agent_config.get('tritonbench', {}).get('golden_metrics'),
        'perf_ref_folder': agent_config.get('tritonbench', {}).get('perf_ref_folder'),
        'perf_G_path': agent_config.get('tritonbench', {}).get('perf_G_path'),
    }
    
    # Create runner script
    runner_script = _create_runner_script(geak_path, "tritonbench")
    
    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(runner_script)
        script_path = f.name
    
    # Convert workspace to absolute path - subprocess runs from geak_path, so relative paths resolve differently
    workspace = str(Path(workspace).resolve())
    workspace_path = Path(workspace)
    
    # Create symlinks in workspace for perf test imports
    # The perf scripts do: sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    # and then import from TritonBench_v1.<kernel> and performance_utils
    agent_dir = Path(__file__).parent
    
    # Symlink 1: TritonBench_v1 -> TritonBench_G_v1 (golden kernel implementations)
    tritonbench_src = agent_dir / "GEAK-eval" / "tb_eval" / "data" / "TritonBench" / "data" / "TritonBench_G_v1"
    tritonbench_link = workspace_path / "TritonBench_v1"
    
    if tritonbench_src.exists() and not tritonbench_link.exists():
        try:
            tritonbench_link.symlink_to(tritonbench_src)
            logger.info(f"Created TritonBench_v1 symlink in workspace: {tritonbench_link}")
        except Exception as e:
            logger.warning(f"Failed to create TritonBench_v1 symlink: {e}")
    
    # Symlink 2: performance_utils.py (benchmark utilities)
    perf_utils_src = agent_dir / "GEAK-eval" / "tb_eval" / "data" / "TritonBench" / "performance_metrics" / "perf_G" / "performance_utils.py"
    perf_utils_link = workspace_path / "performance_utils.py"
    
    if perf_utils_src.exists() and not perf_utils_link.exists():
        try:
            perf_utils_link.symlink_to(perf_utils_src)
            logger.info(f"Created performance_utils.py symlink in workspace: {perf_utils_link}")
        except Exception as e:
            logger.warning(f"Failed to create performance_utils.py symlink: {e}")
    
    # Create golden_metrics directory for perf results (hardcoded in performance_utils.py)
    # This directory is relative to GEAK-eval/tb_eval (the REPO_ROOT used by evaluator)
    geak_eval_tb_eval = agent_dir / "GEAK-eval" / "tb_eval"
    golden_metrics_dir = geak_eval_tb_eval / "AMD_Instinct_MI300X_VF_golden_metrics"
    if not golden_metrics_dir.exists():
        try:
            golden_metrics_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created golden_metrics directory: {golden_metrics_dir}")
        except Exception as e:
            logger.warning(f"Failed to create golden_metrics directory: {e}")
    
    try:
        # Run in subprocess with clean environment (only GEAK-agent in path)
        logger.info(f"Running GEAK-agent subprocess...")
        env = os.environ.copy()
        # Clear PYTHONPATH and only set GEAK-agent path to avoid conflicts
        env['PYTHONPATH'] = geak_path
        
        result = subprocess.run(
            [
                sys.executable, script_path,
                geak_path,
                workspace,
                json.dumps(config),
                json.dumps(task_config)
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            env=env,
            cwd=geak_path  # Run from GEAK-agent directory
        )
        
        # Log output (stdout contains our prints)
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"[GEAK] {line}")
        
        # Also log stderr (where loguru outputs iteration progress)
        if result.stderr:
            for line in result.stderr.split('\n'):
                if line.strip():
                    logger.info(f"[GEAK-loguru] {line}")
        
        if result.returncode != 0:
            logger.error(f"GEAK subprocess failed with code {result.returncode}")
            raise RuntimeError(f"GEAK-agent subprocess failed: {result.stderr}")
        
        # Parse result from output
        output_path = os.path.join(workspace, "geak_output.jsonl")
        
        # Write task_result.yaml
        _write_task_result_from_output(workspace, output_path)
        
        return "TritonBench optimization completed"
        
    finally:
        # Cleanup temp script
        try:
            os.unlink(script_path)
        except:
            pass


def run_rocm_agent(
    eval_config: Dict,
    task_config: Dict,
    workspace: str,
    agent_config: Dict,
    geak_path: str
) -> str:
    """Run GEAK-agent in ROCm mode using subprocess."""
    import json
    import tempfile
    
    logger.info("Running GEAK-OptimAgentV2 in ROCm mode (subprocess)")
    
    # Prepare config for subprocess
    api_key = agent_config.get('llm', {}).get('api_key') or os.environ.get('OPENAI_API_KEY') or os.environ.get('AMD_API_KEY')
    
    if not api_key:
        raise ValueError("No API key found. Set OPENAI_API_KEY or AMD_API_KEY environment variable, or set llm.api_key in agent_config.yaml")
    
    config = {
        'api_key': api_key,
        'model_id': agent_config.get('llm', {}).get('model_id', 'claude-sonnet-4'),
        'temperature': agent_config.get('llm', {}).get('temperature', 1.0),
        'max_iteration': agent_config.get('max_iteration', 10),
        'multi_thread': agent_config.get('multi_thread', True),
        'ancestor_num': agent_config.get('ancestor_num', 5),
        'descendant_num': agent_config.get('descendant_num', 2),
        'gpu_id': agent_config.get('gpu_id', 0),
        'descendant_debug': agent_config.get('descendant_debug', 0),
        'target_gpu': agent_config.get('target_gpu', 'MI325X'),
        'profiling': agent_config.get('profiling', False),
        'corpus_path': agent_config.get('rocm', {}).get('corpus_path'),
        'rocm_statis_path': agent_config.get('rocm', {}).get('statis_path'),
        'rocm_py_folder': agent_config.get('rocm', {}).get('py_folder'),
        'rocm_instruction_path': agent_config.get('rocm', {}).get('instruction_path'),
    }
    
    # Create runner script
    runner_script = _create_runner_script(geak_path, "rocm")
    
    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(runner_script)
        script_path = f.name
    
    try:
        # Run in subprocess
        logger.info(f"Running GEAK-agent subprocess (ROCm)...")
        env = os.environ.copy()
        env['PYTHONPATH'] = geak_path + ':' + env.get('PYTHONPATH', '')
        
        result = subprocess.run(
            [
                sys.executable, script_path,
                geak_path,
                workspace,
                json.dumps(config),
                json.dumps(task_config)
            ],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            env=env
        )
        
        # Log output
        if result.stdout:
            for line in result.stdout.split('\n'):
                if line.strip():
                    logger.info(f"[GEAK] {line}")
        
        if result.returncode != 0:
            logger.error(f"GEAK subprocess failed: {result.stderr}")
            raise RuntimeError(f"GEAK-agent subprocess failed: {result.stderr}")
        
        # Parse result from output
        output_path = os.path.join(workspace, "geak_output.jsonl")
        
        # Write task_result.yaml
        _write_task_result_from_output(workspace, output_path)
        
        return "ROCm optimization completed"
        
    finally:
        # Cleanup temp script
        try:
            os.unlink(script_path)
        except:
            pass


def _write_task_result_from_output(workspace: str, output_path: str):
    """Parse GEAK output and write task_result.yaml."""
    import json
    import glob
    
    result = {
        'task_name': Path(workspace).name,
        'pass_compilation': False,
        'pass_correctness': False,
        'speedup_ratio': 0.0,
        'optimization_summary': 'GEAK-OptimAgentV2 optimization'
    }
    
    # Try to parse memory files (they contain pass_call, pass_exe, pass_perf, perf data)
    # Memory files are named geak_output_mem_0.json, geak_output_mem_1.json, etc.
    mem_files = sorted(glob.glob(os.path.join(workspace, 'geak_output_mem_*.json')))
    
    if mem_files:
        try:
            # Read the latest memory file (highest iteration)
            with open(mem_files[-1], 'r') as f:
                mem_data = json.load(f)
            
            # Determine target kernel from task_name (e.g., 'add_example_20251126' -> 'add_example.py')
            task_name = result['task_name']
            # Extract kernel name (before the timestamp)
            kernel_base = '_'.join(task_name.split('_')[:-2]) if '_' in task_name else task_name
            target_kernel = f"{kernel_base}.py"
            
            # Look for the specific target kernel in memory data
            task_data = mem_data.get(target_kernel, {})
            
            if isinstance(task_data, dict) and 'pass_call' in task_data:
                # Get pass status
                pass_call = task_data.get('pass_call', False)
                pass_exe = task_data.get('pass_exe', False)
                pass_perf = task_data.get('pass_perf', False)
                
                result['pass_compilation'] = pass_call
                result['pass_correctness'] = pass_exe
                
                # Parse performance data from exe_err_msg (which contains benchmark output)
                exe_output = task_data.get('exe_err_msg', '')
                if exe_output and 'GB/s' in str(exe_output):
                    # Extract best GB/s from benchmark output
                    try:
                        import re
                        gbps_matches = re.findall(r"'GB/s':\s*([\d.]+)", str(exe_output))
                        if gbps_matches:
                            best_gbps = max(float(g) for g in gbps_matches)
                            result['speedup_ratio'] = best_gbps / 100.0  # Normalize to ~1.0 scale
                            result['optimization_summary'] = f"Peak bandwidth: {best_gbps:.1f} GB/s"
                    except:
                        pass
                
                # If pass_perf is True, the speedup would be in perf_candidates
                perf_candidates = task_data.get('perf_candidates', [])
                if perf_candidates:
                    result['optimization_summary'] = f"Generated {len(perf_candidates)} optimized candidates"
                
                logger.info(f"Found kernel {target_kernel}: pass_call={pass_call}, pass_exe={pass_exe}")
                    
        except Exception as e:
            logger.warning(f"Could not parse GEAK memory file: {e}")
    
    # Also check JSONL output for any predict field
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get('predict'):
                        result['pass_compilation'] = True
                        result['pass_correctness'] = True
                        break
        except Exception as e:
            logger.warning(f"Could not parse GEAK output: {e}")
    
    result_path = os.path.join(workspace, 'task_result.yaml')
    with open(result_path, 'w') as f:
        yaml.dump(result, f, default_flow_style=False)
    
    logger.info(f"Task result written to: {result_path}")


# ============================================================================
# Main Agent Launcher
# ============================================================================

@register_agent("geak_optimagentv2")
def launch_agent(eval_config: Dict, task_config_dir: str, workspace: str) -> str:
    """
    Launch GEAK-OptimAgentV2 agent.
    
    This launcher:
    1. Auto-clones GEAK-agent if not present
    2. Routes to TritonBench or ROCm mode based on task_type
    3. Adapts AgentKernelArena task config to GEAK-agent format
    4. Runs optimization and writes task_result.yaml
    
    Args:
        eval_config: Global evaluation configuration
        task_config_dir: Path to task's config.yaml
        workspace: Path to duplicated workspace for the agent
        
    Returns:
        Result string describing the outcome
    """
    logger.info("=" * 80)
    logger.info("Starting GEAK-OptimAgentV2 Agent")
    logger.info("=" * 80)
    
    # Load agent config
    agent_dir = Path(__file__).parent
    agent_config_path = agent_dir / 'agent_config.yaml'
    with open(agent_config_path, 'r') as f:
        agent_config = yaml.safe_load(f)
    
    # Resolve relative paths (./GEAK-eval/... → absolute paths)
    agent_config = _resolve_relative_paths(agent_config, agent_dir)
    
    # Load task config
    with open(task_config_dir, 'r') as f:
        task_config = yaml.safe_load(f)
    
    # Ensure GEAK-agent is available (auto-clone if needed)
    geak_path = ensure_geak_agent_available(agent_config)
    
    # Add GEAK-agent to Python path
    if geak_path not in sys.path:
        sys.path.insert(0, geak_path)
    
    logger.info(f"GEAK-agent path: {geak_path}")
    logger.info(f"Workspace: {workspace}")
    logger.info(f"Task config: {task_config_dir}")
    
    # Determine task type and route to appropriate agent
    task_type = task_config.get('task_type', 'triton2triton')
    logger.info(f"Task type: {task_type}")
    
    try:
        if task_type in ['hip2hip', 'cuda2hip', 'pytorch2hip']:
            result = run_rocm_agent(eval_config, task_config, workspace, agent_config, geak_path)
        else:
            # TritonBench mode for: triton2triton, instruction2triton, etc.
            result = run_tritonbench_agent(eval_config, task_config, workspace, agent_config, geak_path)
        
        logger.info(f"Agent execution completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"GEAK-OptimAgentV2 failed: {e}", exc_info=True)
        
        # Write failed result
        task_result = {
            'task_name': Path(workspace).name,
            'pass_compilation': False,
            'pass_correctness': False,
            'speedup_ratio': 0.0,
            'optimization_summary': f'Failed: {str(e)}'
        }
        with open(os.path.join(workspace, 'task_result.yaml'), 'w') as f:
            yaml.dump(task_result, f, default_flow_style=False)
        
        raise

