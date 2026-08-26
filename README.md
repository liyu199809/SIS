# SIS: Turning Off-Policy Tokens On-Policy

**Official code release for ["Turning Off-Policy Tokens On-Policy: A Plug-in Approach for Improving LLM Alignment"](https://arxiv.org/abs/2607.04728)**

<p align="center">
  <a href="https://arxiv.org/abs/2607.04728"><b>Paper</b></a> |
  <a href="#-quick-start"><b>Quick Start</b></a> |
  <a href="#-key-configuration"><b>Configuration</b></a> |
  <a href="#-citation"><b>Citation</b></a>
</p>

---

## 📰 News

- **[2026/08]** 🔥 Initial code release! Training recipes for math RL (`GRPO` / `CISPO`) and agentic search RL (`GRPO` / `DAPO` / `GSPO`), all with the SIS plug-in.

## 🧠 Introduction

RL post-training for LLMs follows the efficient *"rollout then update"* paradigm, which inevitably produces **off-policy training data**. Standard importance sampling (IS) corrects for this, but token-level ratios compound over long sequences and explode the estimator variance.

**Selective Importance Sampling (SIS)** resolves this by *transferring off-policy tokens to on-policy* so that no correction is needed for them. Inspired by rejection sampling, SIS views the off-policy model as the proposal distribution and runs a **token-level rejection test**:

- **Accepted** tokens are treated as on-policy and receive a **unit importance score**;
- **Rejected** tokens retain the standard IS correction.

SIS is theoretically proven to reduce the gap between token-level and sequence-level off-policy gradient estimators. As a **plug-in**, it only modifies the importance ratio in the policy loss:

- ⚡ Negligible wall-clock overhead;
- 🔌 Combines with a wide range of RL post-training algorithms (GRPO, DAPO, GSPO, CISPO, ...);
- 📈 Consistent improvements on dense and MoE LLMs across math and agent benchmarks, with substantially stronger robustness under off-policy data (e.g., stale rollouts).

## 🛠️ Installation

The codebase uses [verl](https://github.com/volcengine/verl) as a git submodule. We will publish our modified fork (which contains the core SIS implementation) soon; before cloning, make sure the submodule can be resolved.

```bash
git clone --recursive https://github.com/liyu199809/sis.git
cd sis

# UV (recommended)
uv sync && source .venv/bin/activate
uv pip install -e verl
uv pip install -e ".[vllm,search_tool]"
uv pip install "flash-attn==2.8.3" --no-build-isolation
```

Or with conda:

```bash
conda create --name sis python=3.10 && conda activate sis
pip install -e verl
pip install -e ".[vllm,search_tool]"
pip install "flash-attn==2.8.3" --no-build-isolation
```

## 🚀 Quick Start

SIS is enabled by a single switch — `actor_rollout_ref.actor.use_clip_less=True` — on top of any supported policy loss. The scripts below are configured this way out of the box.

### 1. Math RL (no tool use)

```bash
# prepare data (DAPO-Math-17K -> train.parquet, MATH-500/AMC23/AIME24/AIME25 test sets)
python examples/data_preprocess/math_simple_rl.py --local_dir $HOME/verl_data/data/math_simple_rl

# GRPO + SIS
bash examples/train/math_simple_rl/train_qwen3_8b_grpo.sh

# CISPO + SIS
bash examples/train/math_simple_rl/train_qwen3_8b_cispo.sh
```

### 2. Agentic Search RL (multi-turn tool use)

Follow [examples/train/search_r1/README.md](examples/train/search_r1/README.md) to download the wiki-18 corpus and e5 retriever index, then:

```bash
# GRPO + SIS
bash examples/train/search_r1/train_8B_sis.sh

# DAPO + SIS
bash examples/train/search_r1/train_8B_dapo.sh

# GSPO + SIS (sequence-level loss + SIS token correction)
bash examples/train/search_r1/train_8B_gspo.sh
```

Model weights, data paths, and checkpoint directories can be overridden via environment variables (`PREFIX`, `MODEL_NAME`, `DATA_ROOT`, `CKPT_ROOT`) without editing the scripts.

## ⚙️ Key Configuration

| Flag | Default | Description |
|---|---|---|
| `actor_rollout_ref.actor.use_clip_less` | `False` | **Enable SIS.** Replaces the vanilla importance ratio with the SIS-corrected weight (via `offpolicy2onpolicy()`) in the policy loss. |
| `actor_rollout_ref.actor.policy_loss.loss_mode` | `vanilla` | Base policy loss: `vanilla` (GRPO), `gspo`, `cispo`, `clip_cov`, `kl_cov`, `gpg`, `geo_mean`. SIS is implemented for `vanilla`, `gspo`, and `cispo`. |
| `actor_rollout_ref.actor.use_sis_kl` | `False` | Additionally apply the SIS correction to the KL term. |
| `actor_rollout_ref.actor.clip_ratio_low` / `clip_ratio_high` | `None` | Asymmetric clipping bounds used on the IS-corrected (rejected) tokens. |
| `actor_rollout_ref.actor.topk_k` | `100` | Number of top-K logits gathered for the token-level rejection test. |
| `+actor_rollout_ref.actor.log_token_acceptance` | `False` | Log per-step token acceptance rates (the "accept rate" analysis in the paper). |

> **Known limitation**: the current implementation repeatedly communicates several large top-K tensors between GPUs, which limits runtime efficiency. This will be fixed in a future release.

## 🗺️ Code Map

| Component | Location |
|---|---|
| **SIS core** (`offpolicy2onpolicy`, token-level rejection test, SIS-KL) | [`verl/verl/trainer/ppo/core_algos.py`](verl/verl/trainer/ppo/core_algos.py) |
| Policy losses (vanilla / GSPO / CISPO / Clip-Cov / ...) | [`verl/verl/trainer/ppo/core_algos.py`](verl/verl/trainer/ppo/core_algos.py) |
| Top-K log-prob computation, SIS wiring, acceptance logging | [`verl/verl/workers/actor/dp_actor.py`](verl/verl/workers/actor/dp_actor.py) |
| Agent training loop (multi-turn tool calling) | [`verl_tool/`](verl_tool/) |
| Training scripts | [`examples/train/`](examples/train/) |

The training framework is built on [verl](https://github.com/volcengine/verl) and [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool); see their docs for the general architecture (tool server, sync/async rollout design).

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{li2026turning,
  title={Turning Off-Policy Tokens On-Policy: A Plug-in Approach for Improving LLM Alignment},
  author={Li, Yu and Li, Xiuyu and Yi, Mingyang and Wang, Jiaxing and Zhang, Liangxu and Xing, Zhaolong and Chen, Zhen},
  journal={arXiv preprint arXiv:2607.04728},
  year={2026}
}
```

## 🙏 Acknowledgements

- [verl](https://github.com/volcengine/verl) and [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool) for the RL training framework and tool-agent infrastructure.
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) for the agentic search training setup and retrieval corpus.

## 📄 License

This project is released under the [MIT License](LICENSE).
