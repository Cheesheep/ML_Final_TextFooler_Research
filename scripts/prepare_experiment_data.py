from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_call_benchmark.augmentations import emotion_rewrite, trust_building_rewrite, urgency_rewrite
from fraud_call_benchmark.data import count_dialog_turns, load_attack_pairs


def normalize_lccc_utterance(text: str) -> str:
    # LCCC 原始对话里空白和换行比较杂，这里先压平，后面统一按 left/right 组织。
    return "".join(str(text).split())


def format_dialog(dialog: list[str]) -> str:
    lines = []
    for idx, utterance in enumerate(dialog):
        speaker = "left" if idx % 2 == 0 else "right"
        lines.append(f"{speaker}: {normalize_lccc_utterance(utterance)}")
    return "\n".join(lines)


def stream_negative_candidates(files: list[Path], target_pool: int = 20000) -> list[dict]:
    # 先从公开对话里捞一批“像正常通话”的候选，再在后面按统计特征精筛。
    candidates: list[dict] = []
    seen: set[str] = set()

    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                dialog = json.loads(line)
                if not isinstance(dialog, list):
                    continue
                if len(dialog) < 6 or len(dialog) > 14:
                    continue
                formatted = format_dialog(dialog)
                if formatted in seen:
                    continue
                turns = count_dialog_turns(formatted)
                chars = len(formatted)
                if turns < 6 or turns > 14:
                    continue
                if chars < 120 or chars > 900:
                    continue
                # 这类文本和诈骗话术过于接近，直接混进负类会把边界搞脏。
                if "退款" in formatted and "链接" in formatted and "银行卡" in formatted:
                    continue
                candidates.append(
                    {
                        "text": formatted,
                        "label": 0,
                        "source": "LCCC",
                        "num_turns": turns,
                        "num_chars": chars,
                    }
                )
                seen.add(formatted)
                if len(candidates) >= target_pool:
                    return candidates
    return candidates


def build_test_variant(base_test: pd.DataFrame, fraud_rewriter, name: str) -> pd.DataFrame:
    # 这里只改正类文本，负类保持不动，方便看模型对同一批欺诈样本改写前后的变化。
    variant = base_test.copy()
    fraud_mask = variant["label"] == 1
    variant.loc[fraud_mask, "text"] = variant.loc[fraud_mask, "text"].map(fraud_rewriter)
    variant["variant"] = name
    return variant


def main() -> None:
    parser = argparse.ArgumentParser(description="构建虚假通话检测实验数据集")
    parser.add_argument(
        "--attack-xlsx",
        default=str(PROJECT_ROOT / "data" / "TextFooler攻击后新的欺诈通话数据.xlsx"),
        help="课程提供的 TextFooler 攻击 Excel",
    )
    parser.add_argument(
        "--lccc-dir",
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="LCCC 原始 jsonl.gz 文件目录",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "fraud_benchmark"),
        help="输出目录",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.random_state)
    # 正类直接来自原始欺诈数据，pair_id 保留下来，后面做误判追踪要用。
    positives = load_attack_pairs(args.attack_xlsx).copy()
    positives["pair_id"] = np.arange(len(positives))
    positives["text"] = positives["original_text"]
    positives["label"] = 1
    positives["source"] = "course_fraud"
    positives["num_turns"] = positives["text"].map(count_dialog_turns)
    positives["num_chars"] = positives["text"].map(len)

    lccc_dir = Path(args.lccc_dir)
    lccc_files = [
        lccc_dir / "lccc_base_train.jsonl.gz",
        lccc_dir / "lccc_base_valid.jsonl.gz",
        lccc_dir / "lccc_base_test.jsonl.gz",
    ]
    for path in lccc_files:
        if not path.exists():
            raise FileNotFoundError(f"未找到 LCCC 文件: {path}")
    negative_candidates = pd.DataFrame(stream_negative_candidates(lccc_files))
    if len(negative_candidates) < len(positives):
        raise ValueError(
            f"LCCC 过滤后候选不足，当前 {len(negative_candidates)}，需要至少 {len(positives)}。"
        )
    positive_turn_mean = float(positives["num_turns"].mean())
    positive_char_mean = float(positives["num_chars"].mean())
    # 负类不是随机抽，尽量往正类的轮次和长度分布上靠，不然太容易被模型学成“长短分类器”。
    negative_candidates["match_score"] = (
        (negative_candidates["num_turns"] - positive_turn_mean).abs() / positive_turn_mean
        + (negative_candidates["num_chars"] - positive_char_mean).abs() / positive_char_mean
    )
    negatives = (
        negative_candidates.sort_values(["match_score", "num_chars"], ascending=[True, False])
        .head(len(positives))
        .reset_index(drop=True)
    )
    negatives["pair_id"] = -1
    negatives["original_text"] = negatives["text"]
    negatives["attacked_text"] = negatives["text"]

    full = pd.concat(
        [
            positives[
                ["pair_id", "text", "label", "source", "num_turns", "num_chars", "original_text", "attacked_text"]
            ],
            negatives[
                ["pair_id", "text", "label", "source", "num_turns", "num_chars", "original_text", "attacked_text"]
            ],
        ],
        ignore_index=True,
    )

    # 这里先做 train/test，再从 train_val 切出 val，保证三份数据都是分层抽样。
    train_val, test = train_test_split(
        full,
        test_size=0.2,
        stratify=full["label"],
        random_state=args.random_state,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.125,
        stratify=train_val["label"],
        random_state=args.random_state,
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()
    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    full_dataset = pd.concat([train, val, test], ignore_index=True)

    original_test = test[["text", "label", "source", "pair_id"]].copy()
    original_test["variant"] = "original"

    # TextFooler 版本复用同一批测试样本，只替换正类文本，避免引入额外样本波动。
    textfooler_test = test.copy()
    fraud_mask = textfooler_test["label"] == 1
    textfooler_test.loc[fraud_mask, "text"] = textfooler_test.loc[fraud_mask, "attacked_text"]
    textfooler_test = textfooler_test[["text", "label", "source", "pair_id"]].copy()
    textfooler_test["variant"] = "textfooler"

    trust_test = build_test_variant(original_test, trust_building_rewrite, "trust")
    urgency_test = build_test_variant(original_test, urgency_rewrite, "urgency")
    emotion_test = build_test_variant(original_test, emotion_rewrite, "emotion")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理后的数据包单独落盘，训练脚本只认这里，不再回头扫原始文件。
    full_dataset.to_csv(output_dir / "full_dataset.csv", index=False, encoding="utf-8-sig")
    original_test.to_csv(output_dir / "original_test.csv", index=False, encoding="utf-8-sig")
    textfooler_test.to_csv(output_dir / "textfooler_test.csv", index=False, encoding="utf-8-sig")
    trust_test.to_csv(output_dir / "trust_test.csv", index=False, encoding="utf-8-sig")
    urgency_test.to_csv(output_dir / "urgency_test.csv", index=False, encoding="utf-8-sig")
    emotion_test.to_csv(output_dir / "emotion_test.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "positive_count": int((full_dataset["label"] == 1).sum()),
        "negative_count": int((full_dataset["label"] == 0).sum()),
        "split_sizes": {
            "train": int((full_dataset["split"] == "train").sum()),
            "val": int((full_dataset["split"] == "val").sum()),
            "test": int((full_dataset["split"] == "test").sum()),
        },
        "positive_avg_turns": round(float(positives["num_turns"].mean()), 3),
        "positive_avg_chars": round(float(positives["num_chars"].mean()), 3),
        "negative_avg_turns": round(float(negatives["num_turns"].mean()), 3),
        "negative_avg_chars": round(float(negatives["num_chars"].mean()), 3),
        "negative_source": "LCCC base subset",
        "random_state": args.random_state,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
