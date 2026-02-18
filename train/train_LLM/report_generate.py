from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from tqdm import tqdm
import torch
from openai import OpenAI
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_TEST_JSON = WORKSPACE / "abnormal_test.json"
DEFAULT_OUTPUT_JSON = WORKSPACE / "results" / "report_results.json"
DEFAULT_BASE_MODEL = WORKSPACE / "merge"
DEFAULT_GRPO_LORA = WORKSPACE / "grpo_lora"

API_SECRET_KEY = "sk-zk26303c340d3fddff99cb1c5e29cb1fcf065c1d4ec9fba7"
BASE_URL = "https://api.zhizengzeng.com/v1/"

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


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def chat_completions4(query: str) -> str:
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
    )
    return resp.choices[0].message.content


def load_grpo_model(base_model: Path, lora_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(lora_dir))
    model.eval()
    return model, tokenizer


def generate_quick_diagnosis(
    model,
    tokenizer,
    ecg_prompt: str,
    max_new_tokens: int = 128,
) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ecg_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def build_report_prompt(
    ecg_prompt: str,
    quick_diagnosis: str,
) -> str:
    return (
        "你是资深心电图医生。已确定患者存在右室肥大。\n"
        "请整合以下ECG信息，结合小模型给出的快速诊断结论，生成完整报告。\n\n"
        "要求：\n"
        "1) 先给出快速诊断结论(可直接引用小模型结论)。\n"
        "2) 对右室肥大及心电图异常进行分析，解释关键指标。\n"
        "3) 输出结构化报告，包含: 诊断结论、诊断分析、建议。\n"
        "4) 内容要简洁、医学表述规范。\n\n"
        "输出格式示例：\n"
        "# 心电图检查报告\n"
        "## 诊断结论\n"
        "- ...\n"
        "## 诊断分析\n"
        "- ...\n"
        "## 建议\n"
        "- ...\n\n"
        "ECG信息:\n"
        f"{ecg_prompt}\n\n"
        "小模型快速诊断结论:\n"
        f"{quick_diagnosis}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--base_model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--grpo_lora", type=Path, default=DEFAULT_GRPO_LORA)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    records = load_records(args.test_json)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    model, tokenizer = load_grpo_model(args.base_model, args.grpo_lora)

    ensure_dir(args.output_json)
    with args.output_json.open("w", encoding="utf-8") as f:
        f.write("[")
        first = True
        for row in tqdm(records, desc="Generating predictions"):
            items = row.get("ecginfo", [])
            ecg_prompt = build_prompt(items)
            quick = generate_quick_diagnosis(
                model,
                tokenizer,
                ecg_prompt,
                max_new_tokens=args.max_new_tokens,
            )
            report_prompt = build_report_prompt(ecg_prompt, quick)
            report = chat_completions4(report_prompt)

            record = {
                "filename": row.get("filename"),
                "ecg_prompt": ecg_prompt,
                "quick_diagnosis": quick,
                "report_prompt": report_prompt,
                "report": report,
            }
            if not first:
                f.write(",\n")
            first = False
            json.dump(record, f, ensure_ascii=False, indent=2)
            f.flush()
        f.write("]\n")


if __name__ == "__main__":
    main()
