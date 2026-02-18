from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Iterable

from openai import OpenAI


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_INPUT_JSON = WORKSPACE / "abnormal_text.json"
DEFAULT_OUTPUT_JSONL = WORKSPACE / "abnormal_experience_sft.jsonl"

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


def write_jsonl(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_experience_template(evidence_lines: list[str], conclusion: str) -> str:
    cleaned = [line.strip() for line in evidence_lines if line.strip()]
    evidence_block = "\n".join(cleaned) if cleaned else "- Evidence of 诊断信息不足"
    return (
        "<experience>:\n"
        f"{evidence_block}\n"
        "</experience>\n"
        f"<conclusion>{conclusion}</conclusion>"
    )


def call_llm(
    client: OpenAI,
    model: str,
    ecg_prompt: str,
    diagnosis: str,
    retries: int = 3,
    sleep_sec: float = 2.0,
) -> list[str]:
    user_prompt = (
        "You are generating training data.\n"
        "Given ECG parameters and a diagnosis, output 3-6 bullet points.\n"
        "Each line must start with '- Evidence of ' and be short.\n"
        "Do not include conclusion or extra text.\n\n"
        f"Diagnosis: {diagnosis}\n\n"
        f"{ecg_prompt}"
    )

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            lines = [line for line in content.splitlines() if line.strip()]
            return lines
        except Exception:
            if attempt == retries:
                return ["- Evidence of 生成失败，需人工复核"]
            time.sleep(sleep_sec * attempt)

    return ["- Evidence of 生成失败，需人工复核"]


def build_sft_dataset(
    input_json: Path,
    output_jsonl: Path,
    model: str,
    api_key: str,
    base_url: str,
    sample_size: int,
    seed: int,
) -> int:
    raw = load_records(input_json)
    filtered = []
    for row in raw:
        diagnosis = (row.get("diagnosis_conclusion") or "").strip()
        items = row.get("ecginfo", [])
        if not diagnosis or not items:
            continue
        filtered.append(row)

    random.Random(seed).shuffle(filtered)
    picked = filtered[: min(sample_size, len(filtered))]

    client = OpenAI(api_key=api_key, base_url=base_url)

    out: list[dict[str, object]] = []
    for row in picked:
        diagnosis = (row.get("diagnosis_conclusion") or "").strip()
        items = row.get("ecginfo", [])
        prompt = build_prompt(items)
        evidence_lines = call_llm(client, model, prompt, diagnosis)
        assistant_content = build_experience_template(evidence_lines, diagnosis)

        out.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
        )

    write_jsonl(out, output_jsonl)
    return len(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=Path,
        default=DEFAULT_INPUT_JSON,
        help="Source JSON with ecginfo and diagnosis_conclusion.",
    )
    parser.add_argument(
        "--output_jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help="Output JSONL for SFT.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Number of records to sample (if available).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="Model name for API.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="sk-zk26303c340d3fddff99cb1c5e29cb1fcf065c1d4ec9fba7",
        help="API key (or set OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.zhizengzeng.com/v1/"),
        help="API base URL.",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Set --api_key or OPENAI_API_KEY.")

    count = build_sft_dataset(
        input_json=args.input_json,
        output_jsonl=args.output_jsonl,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"Wrote {count} records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
