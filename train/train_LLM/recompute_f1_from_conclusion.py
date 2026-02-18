from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def extract_conclusion_tag(text: str) -> str:
    match = re.search(r"<conclusion>(.*?)</conclusion>", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip()


def split_diagnosis(text: str) -> list[str]:
    parts = re.split(r"[\n\r]+", text)
    items = []
    for part in parts:
        part = part.strip()
        if part:
            items.append(part)
    return items


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


def recompute_scores(data: dict[str, object]) -> dict[str, object]:
    samples = data.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("Invalid results file: 'samples' must be a list.")

    has_cover = any("cover_exact_match" in s for s in samples if isinstance(s, dict))

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        prediction = str(sample.get("prediction", ""))
        reference = str(sample.get("reference", ""))
        pred_text = extract_conclusion_tag(prediction)

        pred_items = split_diagnosis(pred_text)
        gold_items = split_diagnosis(reference)

        sample["f1"] = f1_score(pred_items, gold_items)
        if has_cover:
            sample["cover_exact_match"] = cover_exact_match(pred_items, gold_items)

    data["count"] = len(samples)
    data["avg_f1"] = (
        sum(s.get("f1", 0.0) for s in samples if isinstance(s, dict))
        / max(len(samples), 1)
    )
    if has_cover:
        data["avg_cover_exact_match"] = (
            sum(s.get("cover_exact_match", 0.0) for s in samples if isinstance(s, dict))
            / max(len(samples), 1)
        )
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=Path,
        default=Path("results2/abnormal_test_results.json"),
        help="Results JSON to update in-place.",
    )
    args = parser.parse_args()

    input_path = args.input_json
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    updated = recompute_scores(data)

    with input_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
