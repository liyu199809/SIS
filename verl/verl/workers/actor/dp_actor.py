# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from sympy.ntheory import dra
"""
Single Process Actor
"""

import json
import logging
import os
from collections import Counter

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty, calculate_sis_kl
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger, marked_timer
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1
        self.use_clip_less = self.config.get("use_clip_less", False)
        self.use_sis_kl = self.config.get("use_sis_kl", False)
        # Number of top-K logits kept for SIS (clip-less) and SIS-KL.
        # Configurable via actor.topk_k (default 100, matching prior behavior).
        self.topk_k = self.config.get("topk_k", 100)

        # SIS forward-compute profiling (coarse). Off by default.
        # NOTE: marked_timer does NOT call cuda.synchronize(), so timings are
        # async-launch approximations (kernel launch cost, not full GPU time).
        # Sufficient for relative attribution of SIS-specific compute overhead.
        self.sis_compute_profiling = self.config.get("sis_compute_profiling", False)
        # Mirror the toggle into core_algos so offpolicy2onpolicy() can time itself
        # without needing actor config access (it's a plain function). Only the Actor
        # drives SIS loss compute; guarding on role prevents a Ref instance
        # (ref_in_actor=True, sharing the process) from clobbering the toggle.
        if actor_optimizer is not None:
            from verl.trainer.ppo import core_algos as _core_algos

            _core_algos.SIS_PROFILING_ENABLED = self.sis_compute_profiling
        # Accumulator populated inside _forward_micro_batch via marked_timer,
        # flushed to metrics at the end of compute_log_prob / update_policy.
        self._sis_timing_raw: dict[str, float] = {}

        self.log_token_acceptance = self.config.get("log_token_acceptance", False)
        self.token_acceptance_dir = self.config.get("token_acceptance_dir", None)
        self.token_acceptance_log_first_epoch_only = self.config.get("token_acceptance_log_first_epoch_only", True)
        if self.config.get("use_kl_loss", False):
            self.use_sis_kl = False
            print("#########\nuse_sis_kl=False, because use_kl_loss=True\n#########")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _sis_timer(self, name: str):
        """Context manager for SIS compute profiling. No-op when disabled.

        Wraps marked_timer so timings accumulate into self._sis_timing_raw only
        when sis_compute_profiling is enabled. Does NOT call cuda.synchronize()
        (async-launch approximation).
        """
        if self.sis_compute_profiling:
            return marked_timer(name, self._sis_timing_raw)
        from contextlib import nullcontext

        return nullcontext()

    def _flush_sis_timing(self, metrics: dict) -> None:
        """Flush accumulated SIS timing into metrics dict, then reset.

        Merges core_algos.SIS_LOSS_TIMING (offpolicy2onpolicy, populated during
        update_policy) into the local accumulator, then emits all SIS timings
        under actor/*_time keys. Always clears both accumulators to avoid
        cross-step accumulation, even when profiling is disabled.
        """
        from verl.trainer.ppo import core_algos as _core_algos

        # Merge module-level loss timing into local accumulator.
        for k, v in _core_algos.SIS_LOSS_TIMING.items():
            self._sis_timing_raw[k] = self._sis_timing_raw.get(k, 0.0) + v
        _core_algos.SIS_LOSS_TIMING.clear()

        if not self.sis_compute_profiling or not self._sis_timing_raw:
            self._sis_timing_raw = {}
            return
        for k, v in self._sis_timing_raw.items():
            # seconds, already summed across micro-batches
            metrics[f"actor/{k}_time"] = v
        self._sis_timing_raw = {}

    def _accumulate_token_acceptance_counts(
        self,
        responses: torch.Tensor,
        response_mask: torch.Tensor,
        accept_mask: torch.Tensor,
        accepted_counts: Counter,
        rejected_counts: Counter,
    ) -> tuple[int, int]:
        valid_mask = response_mask.bool()
        accepted_mask = accept_mask.bool() & valid_mask
        rejected_mask = (~accepted_mask) & valid_mask

        accepted_ids = responses[accepted_mask].detach().to("cpu").tolist()
        rejected_ids = responses[rejected_mask].detach().to("cpu").tolist()

        accepted_counts.update(int(token_id) for token_id in accepted_ids)
        rejected_counts.update(int(token_id) for token_id in rejected_ids)
        return len(accepted_ids), len(rejected_ids)

    def _flush_token_acceptance_counts(
        self,
        accepted_counts: Counter,
        rejected_counts: Counter,
        accepted_total: int,
        rejected_total: int,
        global_step: int,
    ) -> None:
        if accepted_total + rejected_total == 0 or not self.token_acceptance_dir:
            return

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        os.makedirs(self.token_acceptance_dir, exist_ok=True)
        output_path = os.path.join(self.token_acceptance_dir, f"rank_{rank}.jsonl")
        record = {
            "step": int(global_step),
            "rank": int(rank),
            "accepted_total": int(accepted_total),
            "rejected_total": int(rejected_total),
            "accepted": [[int(token_id), int(count)] for token_id, count in accepted_counts.items()],
            "rejected": [[int(token_id), int(count)] for token_id, count in rejected_counts.items()],
        }
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False,
        return_topk: bool = False, topk_k : int = 100, calculate_logits: bool = False, logits_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            clip_less_out: dict containing clip less outputs
        """
        
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                    if return_topk or calculate_logits:
                        raise ValueError("clip_less is not supported with fused kernels")
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    lse_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True) # (total_nnz, 1)
                    if return_topk:
                        with self._sis_timer("sis/topk"):
                            topk_values_rmpad, topk_indices_rmpad = torch.topk(
                                logits_rmpad, k=topk_k, dim=-1, sorted=False
                            )
                            topk_values_rmpad = topk_values_rmpad - lse_rmpad

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if return_topk:
                        with self._sis_timer("sis/topk_sp_gather"):
                            topk_values_rmpad = gather_outputs_and_unpad(
                                topk_values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                            )
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                            )
                    if calculate_logits:
                        logits_rmpad = gather_outputs_and_unpad(
                            logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                        lse_rmpad = gather_outputs_and_unpad(
                            lse_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )                                                
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if return_topk:
                    with self._sis_timer("sis/topk_pad"):
                        full_topk_values = pad_input(
                            hidden_states=topk_values_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                if calculate_logits:
                    num_valid_tokens = logits_rmpad.shape[0]
                    tracker = torch.arange(1, num_valid_tokens + 1, device=logits_rmpad.device).unsqueeze(-1)

                    padded_tracker = pad_input(
                        hidden_states=tracker,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    ) # 形状: [batch_size, seqlen, 1]

                    # 形状: [batch_size, response_length]
                    response_tracker = padded_tracker[:, -response_length - 1 : -1, 0] - 1 

                    valid_mask = response_tracker >= 0
                    safe_row_ids = response_tracker.clamp(min=0).view(-1) 
                    
                    # logits_indices，形状为 [B, R, K, 1]
                    K = logits_indices.shape[-2]
                    flat_col_indices = logits_indices.view(-1, K)
                    flat_row_indices = safe_row_ids.unsqueeze(-1).expand(-1, K)

                    response_lse_flat = lse_rmpad[safe_row_ids] # LSE [B * R, 1]
                    with self._sis_timer("sis/gather_logits"):
                        requested_logits_flat = logits_rmpad[flat_row_indices, flat_col_indices] - response_lse_flat

                        requested_logits_flat = requested_logits_flat * valid_mask.view(-1).unsqueeze(-1)
                        requested_logits = requested_logits_flat.view(batch_size, response_length, K)
                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if return_topk:
                    topk_logits = full_topk_values[:, -response_length - 1 : -1, :]    # (bsz, response_len, K)
                    topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]  # (bsz, response_len, K)
            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                    if self.use_clip_less:
                        raise ValueError("use_clip_less=True is not supported when use_fused_kernels=True")

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)

                    lse = torch.logsumexp(logits, dim=-1, keepdim=True) # (bsz, response_length, 1)
                    if return_topk:
                        with self._sis_timer("sis/topk"):
                            topk_logits, topk_indices = torch.topk(logits, k=topk_k, dim=-1)
                            topk_logits = topk_logits - lse # (bsz, response_len, K)
                    if calculate_logits:
                        with self._sis_timer("sis/gather_logits"):
                            requested_logits = torch.gather(logits, dim=-1, index=logits_indices)
                            requested_logits = requested_logits - lse

                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
            
            clip_less_out = {}
            if return_topk:
                clip_less_out["topk_logits"] = topk_logits
                clip_less_out["topk_indices"] = topk_indices
            if calculate_logits:
                clip_less_out["requested_logits"] = requested_logits
                
            return entropy, log_probs, clip_less_out

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False, return_topk=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)
        
        log_probs_lst = []
        entropy_lst = []
        topk_logits_lst = []
        topk_indices_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, clip_less_out = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy, return_topk=return_topk,
                    topk_k=self.topk_k
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if return_topk:
                topk_logits_lst.append(clip_less_out["topk_logits"])
                topk_indices_lst.append(clip_less_out["topk_indices"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if return_topk:
            with self._sis_timer("sis/topk_concat"):
                out = {
                    "topk_logits": torch.concat(topk_logits_lst, dim=0),
                    "topk_indices": torch.concat(topk_indices_lst, dim=0),
                }
        else:
            out = {}

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys, out

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        
        if self.use_clip_less:
            select_keys.append("topk_logits")
            select_keys.append("topk_indices")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        token_acceptance_enabled = bool(self.log_token_acceptance and self.use_clip_less and self.token_acceptance_dir)
        token_acceptance_accepted_counts = Counter()
        token_acceptance_rejected_counts = Counter()
        token_acceptance_accepted_total = 0
        token_acceptance_rejected_total = 0

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if self.log_token_acceptance and not self.token_acceptance_dir and rank == 0:
            logger.warning("log_token_acceptance=True but token_acceptance_dir is not set; skipping token logs.")

        for epoch_idx in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    if self.use_clip_less:
                        old_log_prob_topk = model_inputs["topk_logits"].squeeze(-1)
                        old_log_prob_indices = model_inputs["topk_indices"]
                    else:
                        old_log_prob_topk, old_log_prob_indices = None, None

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    
                    if self.use_clip_less:
                        entropy, log_prob, clip_less_out = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            calculate_logits=True, logits_indices=old_log_prob_indices,
                            topk_k=self.topk_k
                        )
                        log_prob_topk = clip_less_out["requested_logits"]
                    else:
                        entropy, log_prob, clip_less_out = self._forward_micro_batch(
                            model_inputs, temperature=temperature, calculate_entropy=calculate_entropy,
                            topk_k=self.topk_k
                        )
                        log_prob_topk = None

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout importance sampling weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    
                    # Compute policy loss
                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, accept_rate, D_original, D_sis, accept_mask_out = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        old_log_prob_topk=old_log_prob_topk,
                        log_prob_topk=log_prob_topk, 
                    )
                    micro_batch_metrics["actor/accept_rate"] = accept_rate

                    should_log_token_acceptance = token_acceptance_enabled and accept_mask_out is not None
                    if self.token_acceptance_log_first_epoch_only:
                        should_log_token_acceptance = should_log_token_acceptance and epoch_idx == 0
                    if should_log_token_acceptance:
                        accepted_total, rejected_total = self._accumulate_token_acceptance_counts(
                            responses=model_inputs["responses"],
                            response_mask=response_mask,
                            accept_mask=accept_mask_out,
                            accepted_counts=token_acceptance_accepted_counts,
                            rejected_counts=token_acceptance_rejected_counts,
                        )
                        token_acceptance_accepted_total += accepted_total
                        token_acceptance_rejected_total += rejected_total

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    if self.use_sis_kl:
                        if log_prob_topk is None or old_log_prob_topk is None:
                            raise ValueError("log_prob_topk and old_log_prob_topk must be provided when use_sis_kl is True")
                        
                        sis_kl_loss = calculate_sis_kl(
                            log_prob_topk, old_log_prob_topk, response_mask
                        )
                        policy_loss = policy_loss + sis_kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = sis_kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                            "actor/accept_rate": accept_rate,
                            "actor/D_original": D_original,
                            "actor/D_sis": D_sis,
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()

        if token_acceptance_enabled:
            self._flush_token_acceptance_counts(
                accepted_counts=token_acceptance_accepted_counts,
                rejected_counts=token_acceptance_rejected_counts,
                accepted_total=token_acceptance_accepted_total,
                rejected_total=token_acceptance_rejected_total,
                global_step=data.meta_info.get("global_steps", -1),
            )
            append_to_dict(
                metrics,
                {
                    "actor/token_acceptance_accepted_total": token_acceptance_accepted_total,
                    "actor/token_acceptance_rejected_total": token_acceptance_rejected_total,
                },
            )
        # Flush SIS compute profiling (covers both compute_log_prob and update_policy
        # phases of this step, since the same dp_actor instance persists across calls).
        self._flush_sis_timing(metrics)
        return metrics
