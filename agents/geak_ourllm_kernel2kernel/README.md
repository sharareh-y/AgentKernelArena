## Overview
This agent enables you to evaluation self-trained LLMs via vllm.

## Docker
You need to enter a docker that support VLLM. e.g. rlsys/rocm-6.3.4-patch:rocm6.3.4-numa-patch_ubuntu-22.04

## Downloads & install
- Clone this specific [GEAK_HIP](https://github.com/AMD-AGI/GEAK_HIP.git) repo into this folder and rename it to GEAK-agent. Other version of GEAK-HIP is currently not supported. git clone https://github.com/AMD-AGI/GEAK_HIP.git
- run install.sh in the root folder. It is ok if apt-get install gawk got error.

## Set API
Notice that you should set up a VLLM api first, like this
export VLLM_DISABLE_QUANTIZATION=1
model_path=YOUR MODEL PATH
model_name=YOU CAN WHATEVER NAME YOUR MODEL
"HIP_VISIBLE_DEVICES=0 vllm serve ${model_path} --served-model-name ${model_name}  --host 0.0.0.0 --port 8001 --trust-remote-code --tensor-parallel-size 1 --quantization None --max-num-batched-tokens 65536  --gpu-memory-utilization 0.9 --max-model-len 40960"

## Write Agent Config
- The model name in agent config should be identical to the one you set in your API
- the seqlence length is suggested to firstly set to 16384, which will be divided by 2 if reaches the max length of your model
- descendant and iter are suggested to set to 4 and 10 respectively. If you want to quickly run the evaluation process, you can set to 2 and 2 respectively, but not so precise.

## TODO

- **Decouple GPU architecture from prompt**: `launch_agent.py:104-105` hardcodes
  MI250 specs ("208KB LDS per CU, 64 CUs total") directly in the prompt string.
  This should read from `agent_config.yaml` `target_gpu` and load the corresponding
  architecture spec from `src/prompts/cheatsheet/default_cheatsheet.yaml` (the
  `architecture` section) instead of embedding hardware numbers in code.