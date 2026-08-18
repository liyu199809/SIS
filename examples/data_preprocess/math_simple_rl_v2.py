"""
Preprocess math datasets for simple RL training with \\boxed{} format.

V2 differences vs math_simple_rl.py:
  - All val benchmarks renamed so their data_source matches the
    `startswith("qwen")` branch in verl/utils/reward_score/__init__.py,
    which dispatches to `math_reward.compute_score` (Hendrycks original:
    full-response \\boxed{} extraction + strip_string normalization + ==).
        MATH-500 : math_dapo  -> qwen_math500
        AMC23    : math_amc23 -> qwen_amc2023
        AIME24   : math_aime24-> qwen_aime_2024
        AIME25   : math_aime25-> qwen_aime_2025
    This is the reward path used by the 80+ MATH-500 wandb run
    (sheriyuo-team/sis/cmzlfnt6), and it does NOT require a specific
    prompt template — it only requires the response to contain a
    \\boxed{...} somewhere, which the current prompt already asks for.
  - Train data_source kept as "math_dapo" so training reward still goes
    through math_dapo.compute_score (default minerva, no strict_box_verify).
  - Output written to a separate directory so existing run is untouched.

Sources unchanged:
  - Train: DAPO-Math-17k
  - Test:  MATH-500, AMC23 (math-ai/amc23),
           AIME24 (Maxwell-Jia/AIME_2024),
           AIME25 (opencompass/AIME2025, parts I+II)
"""
import os
import argparse
import datasets


simple_rl_system_prompt = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
SUFFIX = "\n\nRemember to put your answer on its own line after \"Answer:\"."


def extract_question_from_prompt(prompt_content: str) -> str:
    content = prompt_content
    if content.startswith(PREFIX):
        content = content[len(PREFIX):]
    if content.endswith(SUFFIX):
        content = content[:-len(SUFFIX)]
    return content


def _normalize_answer(ans):
    """Convert any answer (float / int / str) to a stripped string.
    For floats that are integers (e.g. 27.0) drop the trailing .0 so they
    match common AMC/AIME ground-truth conventions.
    """
    if isinstance(ans, float):
        if ans.is_integer():
            return str(int(ans))
        return str(ans)
    return str(ans).strip()


def _make_test_record(question: str, answer, idx: int, data_source: str,
                       n_samples: int, extra: dict | None = None):
    extra_info = {"split": "test", "index": idx, "n_samples": n_samples}
    if extra:
        extra_info.update(extra)
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": simple_rl_system_prompt},
            {"role": "user", "content": question},
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": _normalize_answer(answer),
        },
        "extra_info": extra_info,
    }


# ---------- mapping functions ----------
def process_dapo_train(example):
    prompt_content = example["prompt"][0]["content"]
    question = extract_question_from_prompt(prompt_content)

    example["prompt"] = [
        {"role": "system", "content": simple_rl_system_prompt},
        {"role": "user", "content": question},
    ]
    # CHANGED: route training reward through math_reward.compute_score
    # (Hendrycks, full-response \boxed{} extraction). Matches val path
    # so training and validation use the same scorer.
    example["data_source"] = "qwen_dapo_math"
    example["ability"] = "math"
    example["reward_model"] = {
        "style": "rule",
        "ground_truth": example["reward_model"]["ground_truth"],
    }
    return example


def process_math500(example, idx):
    return _make_test_record(
        question=example["problem"],
        answer=example["answer"],
        idx=idx,
        data_source="qwen_math500",     # CHANGED: -> math_reward (Hendrycks)
        n_samples=1,
    )


def process_amc23(example, idx):
    return _make_test_record(
        question=example["question"],
        answer=example["answer"],
        idx=idx,
        data_source="qwen_amc2023",     # CHANGED: -> math_reward (Hendrycks)
        n_samples=32,
    )


def process_aime24(example, idx):
    return _make_test_record(
        question=example["Problem"],
        answer=example["Answer"],
        idx=idx,
        data_source="qwen_aime_2024",   # CHANGED: -> math_reward (Hendrycks)
        n_samples=32,
    )


def process_aime25(example, idx):
    return _make_test_record(
        question=example["question"],
        answer=example["answer"],
        idx=idx,
        data_source="qwen_aime_2025",   # CHANGED: -> math_reward (Hendrycks)
        n_samples=32,
    )


# ---------- driver ----------
def build_train(data_root: str, out_dir: str):
    dapo_path = os.path.join(data_root, "DAPO-Math-17k", "dapo-math-17k.parquet")
    print(f"[train] Loading DAPO-Math-17k from {dapo_path}")
    ds = datasets.load_dataset("parquet", data_files=dapo_path, split="train")
    print(f"[train] {len(ds)} examples")
    ds = ds.map(process_dapo_train, num_proc=8, desc="DAPO-Math-17k")
    out = os.path.join(out_dir, "train.parquet")
    ds.to_parquet(out)
    print(f"[train] saved -> {out}")
    print(f"[train] sample: {ds[0]}\n")


def build_test(data_root: str, out_dir: str):
    # MATH-500
    p = os.path.join(data_root, "MATH-500", "test.jsonl")
    print(f"[test] MATH-500 <- {p}")
    ds = datasets.load_dataset("json", data_files=p, split="train")
    ds = ds.map(process_math500, with_indices=True,
                remove_columns=ds.column_names, num_proc=1,
                desc="MATH-500")
    out = os.path.join(out_dir, "math500_test.parquet")
    ds.to_parquet(out)
    print(f"[test] MATH-500 -> {out}  rows={len(ds)}")

    # AMC23
    p = os.path.join(data_root, "amc23", "test-00000-of-00001.parquet")
    print(f"[test] AMC23 <- {p}")
    ds = datasets.load_dataset("parquet", data_files=p, split="train")
    ds = ds.map(process_amc23, with_indices=True,
                remove_columns=ds.column_names, num_proc=1,
                desc="AMC23")
    out = os.path.join(out_dir, "amc23_test.parquet")
    ds.to_parquet(out)
    print(f"[test] AMC23 -> {out}  rows={len(ds)}")

    # AIME24
    p = os.path.join(data_root, "AIME_2024", "aime_2024_problems.parquet")
    print(f"[test] AIME24 <- {p}")
    ds = datasets.load_dataset("parquet", data_files=p, split="train")
    ds = ds.map(process_aime24, with_indices=True,
                remove_columns=ds.column_names, num_proc=1,
                desc="AIME24")
    out = os.path.join(out_dir, "aime24_test.parquet")
    ds.to_parquet(out)
    print(f"[test] AIME24 -> {out}  rows={len(ds)}")

    # AIME25 (concat I + II)
    p1 = os.path.join(data_root, "AIME2025", "aime2025-I.jsonl")
    p2 = os.path.join(data_root, "AIME2025", "aime2025-II.jsonl")
    print(f"[test] AIME25 <- {p1}, {p2}")
    ds1 = datasets.load_dataset("json", data_files=p1, split="train")
    ds2 = datasets.load_dataset("json", data_files=p2, split="train")
    ds = datasets.concatenate_datasets([ds1, ds2])
    ds = ds.map(process_aime25, with_indices=True,
                remove_columns=ds.column_names, num_proc=1,
                desc="AIME25")
    out = os.path.join(out_dir, "aime25_test.parquet")
    ds.to_parquet(out)
    print(f"[test] AIME25 -> {out}  rows={len(ds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str,
                        default="/mnt/storage/disk1/verl_data/data")
    parser.add_argument("--local_dir", type=str,
                        default="/mnt/storage/disk1/verl_data/data/math_simple_rl_v2")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    if not args.skip_train:
        build_train(args.data_root, args.local_dir)
    if not args.skip_test:
        build_test(args.data_root, args.local_dir)

    print("Done. Outputs in:", args.local_dir)
