# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl-tool is a unified and easy-to-extend tool-agent training framework based on verl (Volcano Engine Reinforcement Learning for LLMs). It enables multi-turn interactive reinforcement learning (RL) for training tool-using agents with capabilities like math reasoning, code generation, web search, and more.

Key features:
- Complete decoupling of actor rollout and environment interaction
- Tool-as-environment paradigm with state persistence
- Native RL framework for tool-calling agents
- User-friendly evaluation suite

## Repository Structure

```
verl-tool/
├── verl/                    # Git submodule - core verl RL framework
├── verl_tool/               # Main verl-tool codebase
│   ├── agent_loop/          # Agent interaction loop logic
│   ├── llm_agent/           # LLM agent with dynamic tool-calling
│   ├── servers/             # Tool server implementations
│   │   └── tools/           # Individual tools (python, search, bash, etc.)
│   ├── trainer/             # Trainer implementations (based on verl)
│   ├── workers/             # Rollout and reward manager workers
│   │   ├── rollout/         # Rollout workers
│   │   └── reward_manager/  # Reward computation for tasks
│   └── utils/               # Utility functions
├── examples/
│   ├── train/               # Training recipes (math_tir, acecoder, search_r1, etc.)
│   └── data_preprocess/     # Data preprocessing scripts
├── benchmarks/              # Evaluation benchmarks
├── assets/docs/             # Documentation files
└── pyproject.toml           # Project dependencies
```

## Installation

### Option 1: UV Installation (Recommended)
```bash
git submodule update --init --recursive
uv sync
source .venv/bin/activate
uv pip install -e verl
uv pip install -e ".[vllm,acecoder,torl,search_tool]"
uv pip install "flash-attn==2.8.3" --no-build-isolation
```

### Option 2: Conda Installation
```bash
git submodule update --init --recursive
conda create --name verl-tool-env python=3.10
conda activate verl-tool-env
pip install -e verl
pip install -e ".[vllm,acecoder,torl,search_tool]"
pip install "flash-attn==2.8.3" --no-build-isolation
```

### Megatron Installation (Optional)
```bash
uv pip install megatron-core
uv pip install --no-build-isolation transformer-engine[pytorch]
```

## Common Development Commands

### Training

Launch training using provided recipe scripts:
```bash
# Math reasoning with GRPO
bash examples/train/math_tir/train_1.5b_grpo.sh

# Math reasoning with DAPO
bash examples/train/math_tir/train_1.5b_dapo.sh

# Search-R1 training
bash examples/train/search_r1/train_7b.sh

# AceCoder training
bash examples/train/acecoder/train_with_tool.sh
```

### Data Preprocessing
```bash
python examples/data_preprocess/deepmath.py --data_source zwhe99/DeepMath-103K --local_dir data/deepmath_torl --sys_prompt_style torl
```

### Code Linting and Formatting
The project uses pre-commit hooks (in `verl/` submodule):
```bash
cd verl
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Key Architecture Concepts

### Synchronous Rollout Design
verl-tool extends verl's `ActorRolloutRefWorker` with `AgentActorRolloutRefWorker` to add tool-calling capabilities. The key modification is overriding the `generate_sequences` function to delegate to `AgentActorManager.run_llm_loop` when agent mode is enabled.

### Tool Server
Tools are implemented in `verl_tool/servers/tools/` with a unified base class. Available tools include:
- `python_code.py` - Python code execution
- `bash_terminal.py` - Bash terminal access
- `google_search.py` / `bing_search.py` - Web search
- `sql.py` - SQL execution
- `text_browser.py` - Web browsing
- `mcp_interface.py` - Model Context Protocol (MCP) integration

### Reward Managers
Reward computation is handled in `verl_tool/workers/reward_manager/`. Each task has its own reward manager that defines how to score model outputs.

## Configuration

Training scripts use Hydra configuration. Key configuration files are in `verl/verl/trainer/config/`. Agent-specific configuration is defined in `AgentActorConfig`.

## Tips for Development

1. **Low VRAM GPUs**: Set `do_offload=True`, `enforce_eager=True`, `tensor_parallel_size=1`, `use_dynamic_bsz=False`, and small `ppo_micro_batch_size_per_gpu`
2. **High VRAM GPUs**: Set `do_offload=False` and `use_dynamic_bsz=True` for faster training
3. **VLLM issues**: Lower `workers_per_tool` and `gpu_memory_utilization` if generation gets stuck
4. **CPU OOM**: Try setting `do_offload=False` and lowering `gpu_memory_utilization`
5. **Large models**: Use `strategy="fsdp"` instead of `fsdp2` if model loading causes OOM on rank 0
6. **Training logs**: Check `verl_step_records/` directory (or `{checkpoint_dir}/step_records` for multi-node) for step-by-step interaction records

## Important Links

- Documentation: `assets/docs/`
  - `install.md` - Installation guide
  - `training_guide.md` - Training guide
  - `sync_design.md` - Synchronous rollout design
  - `asyncRL.md` - Asynchronous rollout design
  - `tool_server.md` - Tool server documentation
  - `evaluation.md` - Evaluation guide
- Paper: https://arxiv.org/abs/2509.01055
- Examples: `examples/train/` - Training recipes for various tasks
