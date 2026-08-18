# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Reward manager that wraps `verl.utils.reward_score.math_dapo.compute_score`.

Plug into training via:
    reward_model.reward_manager=math_dapo

`compute_score` returns: {"score": +1.0 / -1.0, "acc": bool, "pred": str}
"""
import re
import torch
import numpy as np
from collections import defaultdict

from verl import DataProto
from verl.utils.reward_score.math_dapo import compute_score as math_dapo_compute_score
from verl.workers.reward_manager import register


def _extract_last_boxed(text: str) -> str:
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else ""


@register("math_dapo")
class MathDapoRewardManager:
    """Reward manager keyed on `math_dapo.compute_score`.

    Works for any data_source whose ground_truth is a final math answer
    extractable from the model's last \\boxed{...}.
    """
    name = "math_dapo"

    def __init__(self, tokenizer, num_examine, compute_score=None,
                 reward_fn_key="data_source", **kwargs):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        # Always pin to math_dapo.compute_score regardless of the dispatcher
        # the trainer hands us — the default dispatcher requires `data_source`
        # and would fail with `missing 1 required positional argument: 'data_source'`.
        self.compute_score = math_dapo_compute_score
        self.reward_fn_key = reward_fn_key
        # strict_box_verify: require the answer to be in the last \boxed{}.
        # math_dapo defaults to False (Minerva-style normalization).
        self.strict_box_verify = bool(kwargs.get("strict_box_verify", True))

    def __call__(self, data: DataProto, return_dict=False):
        # If a sandbox/RM has already attached scores, just return them.
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {k: data.non_tensor_batch[k] for k in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"],
                        "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_printed = {}

        for i in range(len(data)):
            item = data[i]

            prompt_ids = item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_len = item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            response_ids = item.batch["responses"]
            valid_response_len = item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_len]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch[self.reward_fn_key]

            result = self.compute_score(
                solution_str=response_str,
                ground_truth=ground_truth,
                strict_box_verify=self.strict_box_verify,
            )

            score = float(result["score"])           # +1.0 / -1.0
            acc = 1.0 if result["acc"] else 0.0
            pred = result.get("pred", "") or _extract_last_boxed(response_str)

            scores_i = {"score": score, "acc": acc,
                        "has_answer": 1.0 if pred else 0.0}

            if acc > 0:
                reward_extra_info["correct_response_length"].append(int(valid_response_len))
            else:
                reward_extra_info["wrong_response_length"].append(int(valid_response_len))

            for k, v in scores_i.items():
                reward_extra_info[k].append(v)

            # at validation time (num_examine==1) verl trainer prefers accuracy
            reward = scores_i["acc"] if self.num_examine == 1 else scores_i["score"]
            reward_tensor[i, valid_response_len - 1] = reward

            already_printed.setdefault(data_source, 0)
            if already_printed[data_source] < self.num_examine:
                already_printed[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[pred]", pred)
                for k, v in scores_i.items():
                    print(f"[{k}]", v)

        correct_mean = (np.mean(reward_extra_info["correct_response_length"])
                        if reward_extra_info["correct_response_length"] else None)
        wrong_mean = (np.mean(reward_extra_info["wrong_response_length"])
                      if reward_extra_info["wrong_response_length"] else None)
        reward_extra_info["correct_response_length"] = [correct_mean] * len(reward_tensor)
        reward_extra_info["wrong_response_length"] = [wrong_mean] * len(reward_tensor)

        if return_dict:
            return {"reward_tensor": reward_tensor,
                    "reward_extra_info": dict(sorted(reward_extra_info.items()))}
        return reward_tensor
