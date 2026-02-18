from __future__ import annotations

import argparse
import csv
import json
import os
import re
import inspect
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_RL_JSON = WORKSPACE / "abnormal_rl.json"
DEFAULT_BASE_MODEL = WORKSPACE / "merge"
DEFAULT_OUTPUT_DIR = WORKSPACE / "grpo_lora"
DEFAULT_METRICS_CSV = WORKSPACE / "results" / "grpo_metrics.csv"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_prompt(items: Iterable[dict[str, str]]) -> str:
    lines = ["ECG parameters:"]
    for item in items:
        title = (item.get("title") or "").strip()
        value = (item.get("value") or "").strip()
        if not title and not value:
            continue
        lines.append(f"- {title}: {value}")
    return "\n".join(lines)


def load_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_conclusion_tag(text: str) -> str:
    match = re.search(r"<conclusion>(.*?)</conclusion>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def extract_conclusion_or_text(text: str) -> str:
    extracted = extract_conclusion_tag(text)
    return extracted if extracted else text


def split_diagnosis(text: str) -> list[str]:
    parts = re.split(r"[\n\r]+", text)
    items: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        items.append(part)
    return [item for item in items if item]


def f1_score(pred_items: list[str], gold_items: list[str]) -> float:
    pred_set = set(pred_items)
    gold_set = set(gold_items)
    if not pred_set and not gold_set:
        return 1.0
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    precision = overlap / len(pred_set)
    recall = overlap / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cover_exact_match(pred_items: list[str], gold_items: list[str]) -> float:
    pred_set = set(pred_items)
    gold_set = set(gold_items)
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set:
        return 0.0
    return 1.0 if gold_set.issubset(pred_set) else 0.0


class StepLogger:
    def __init__(self, csv_path: Path) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path = csv_path
        self.step = 0
        self._file = csv_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["step", "reward", "f1", "cover_exact_match"])
        self._file.flush()

    def log(self, reward: float, f1: float, cem: float) -> None:
        self.step += 1
        self._writer.writerow([self.step, reward, f1, cem])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def build_rl_dataset(
    tokenizer: AutoTokenizer, input_json: Path
) -> Dataset:
    raw = load_records(input_json)
    rows: list[dict[str, str]] = []
    for row in raw:
        items = row.get("ecginfo", [])
        diagnosis = (row.get("diagnosis_conclusion") or "").strip()
        if not diagnosis:
            continue
        prompt = build_prompt(items)
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        rows.append(
            {
                "prompt": chat_prompt,
                "reference": diagnosis,
                "filename": row.get("filename", ""),
            }
        )
    return Dataset.from_list(rows)


class RewardComputer:
    __name__ = "f1_reward"

    def __init__(self, logger: StepLogger | None = None) -> None:
        self.logger = logger

    def __call__(self, prompts=None, completions=None, **kwargs):
        samples = kwargs.get("samples")
        references = (
            kwargs.get("references")
            or kwargs.get("reference")
            or kwargs.get("ref")
        )

        if samples is not None and completions is None:
            completions = [s.get("completion", "") for s in samples]
            if references is None:
                references = [s.get("reference", "") for s in samples]

        if completions is None:
            raise ValueError("Reward function did not receive completions.")

        if references is None:
            references = [""] * len(completions)

        if len(references) != len(completions):
            if len(references) > 0 and len(completions) % len(references) == 0:
                repeat = len(completions) // len(references)
                references = [r for r in references for _ in range(repeat)]
            else:
                references = (references + [""] * len(completions))[: len(completions)]

        rewards: list[float] = []
        f1_scores: list[float] = []
        cem_scores: list[float] = []

        for completion, reference in zip(completions, references):
            pred_text = extract_conclusion_tag(completion)
            gold_text = extract_conclusion_or_text(reference)
            pred_items = split_diagnosis(pred_text)
            gold_items = split_diagnosis(gold_text)
            f1 = f1_score(pred_items, gold_items)
            cem = cover_exact_match(pred_items, gold_items)
            rewards.append(f1)
            f1_scores.append(f1)
            cem_scores.append(cem)

        if self.logger is not None:
            mean_reward = sum(rewards) / max(len(rewards), 1)
            mean_f1 = sum(f1_scores) / max(len(f1_scores), 1)
            mean_cem = sum(cem_scores) / max(len(cem_scores), 1)
            self.logger.log(mean_reward, mean_f1, mean_cem)

        return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rl_json",
        type=Path,
        default=DEFAULT_RL_JSON,
        help="RL training JSON with ecginfo and diagnosis_conclusion.",
    )
    parser.add_argument(
        "--base_model",
        type=Path,
        default=DEFAULT_BASE_MODEL,
        help="Merged base model directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for GRPO LoRA.",
    )
    parser.add_argument(
        "--metrics_csv",
        type=Path,
        default=DEFAULT_METRICS_CSV,
        help="CSV path to save per-step reward and metrics.",
    )
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_generations", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_rl_dataset(tokenizer, args.rl_json)

    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    metrics_logger = StepLogger(args.metrics_csv)
    reward_fn = RewardComputer(metrics_logger)

    grpo_kwargs = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "logging_steps": 1,
        "report_to": [],
        "bf16": torch.cuda.is_available(),
        "fp16": not torch.cuda.is_available(),
        "max_new_tokens": args.max_new_tokens,
        "num_generations": args.num_generations,
    }
    sig = inspect.signature(GRPOConfig)
    supported = {k: v for k, v in grpo_kwargs.items() if k in sig.parameters}
    if "max_new_tokens" not in sig.parameters and "max_completion_length" in sig.parameters:
        supported["max_completion_length"] = args.max_new_tokens
    if "num_generations" not in sig.parameters and "num_rollouts" in sig.parameters:
        supported["num_rollouts"] = args.num_generations

    grpo_args = GRPOConfig(**supported)

    trainer_kwargs = {
        "model": model,
        "args": grpo_args,
        "tokenizer": tokenizer,
        "train_dataset": dataset,
        "reward_fn": reward_fn,
        "reward_funcs": [reward_fn],
    }
    trainer_sig = inspect.signature(GRPOTrainer)
    supported_trainer_kwargs = {
        k: v for k, v in trainer_kwargs.items() if k in trainer_sig.parameters
    }
    trainer = GRPOTrainer(**supported_trainer_kwargs)

    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metrics_logger.close()


if __name__ == "__main__":
    main()
