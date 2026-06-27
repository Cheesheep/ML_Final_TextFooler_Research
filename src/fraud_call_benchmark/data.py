from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ATTACK_ORIGINAL_COLUMN = "原始的通话记录"
ATTACK_MUTATED_COLUMN = "textfooler攻击后的通话记录"


@dataclass
class AttackAudit:
    rows: int
    columns: list[str]
    avg_original_chars: float
    avg_attacked_chars: float
    avg_original_turns: float
    avg_attacked_turns: float


@dataclass
class DatasetBundle:
    full_dataset: pd.DataFrame
    original_test: pd.DataFrame
    textfooler_test: pd.DataFrame
    trust_test: pd.DataFrame
    urgency_test: pd.DataFrame
    emotion_test: pd.DataFrame
    metadata: dict


def load_attack_pairs(excel_path: str | Path) -> pd.DataFrame:
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到攻击样本文件: {path}")

    frame = pd.read_excel(path)
    missing = [
        column
        for column in (ATTACK_ORIGINAL_COLUMN, ATTACK_MUTATED_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"攻击样本缺少必要列: {missing}")

    subset = frame[[ATTACK_ORIGINAL_COLUMN, ATTACK_MUTATED_COLUMN]].copy()
    subset.columns = ["original_text", "attacked_text"]
    subset["original_text"] = subset["original_text"].astype(str)
    subset["attacked_text"] = subset["attacked_text"].astype(str)
    return subset


def count_dialog_turns(text: str) -> int:
    return text.count("left:") + text.count("right:")


def audit_attack_pairs(frame: pd.DataFrame) -> AttackAudit:
    return AttackAudit(
        rows=len(frame),
        columns=list(frame.columns),
        avg_original_chars=frame["original_text"].map(len).mean(),
        avg_attacked_chars=frame["attacked_text"].map(len).mean(),
        avg_original_turns=frame["original_text"].map(count_dialog_turns).mean(),
        avg_attacked_turns=frame["attacked_text"].map(count_dialog_turns).mean(),
    )


def load_labelled_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到标注数据集: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix == ".jsonl":
        frame = pd.read_json(path, lines=True)
    elif suffix == ".json":
        frame = pd.read_json(path)
    else:
        raise ValueError(f"不支持的数据格式: {suffix}")

    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(
            "标注数据集必须至少包含 text 和 label 两列，"
            f"当前列为: {list(frame.columns)}"
        )

    normalized = frame.copy()
    normalized["text"] = normalized["text"].astype(str)
    normalized["label"] = normalized["label"].astype(int)
    if normalized["label"].nunique() < 2:
        raise ValueError("标注数据集至少需要包含两个类别，当前无法完成监督学习训练。")
    return normalized


def split_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" in frame.columns:
        train = frame[frame["split"] == "train"].copy()
        val = frame[frame["split"] == "val"].copy()
        test = frame[frame["split"] == "test"].copy()
        if train.empty or test.empty:
            raise ValueError("如果提供 split 列，至少需要包含 train 和 test 两个划分。")
        return train, val, test

    shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
    total = len(shuffled)
    train_end = int(total * 0.7)
    val_end = int(total * 0.85)
    return (
        shuffled.iloc[:train_end].copy(),
        shuffled.iloc[train_end:val_end].copy(),
        shuffled.iloc[val_end:].copy(),
    )


def format_label_distribution(labels: Iterable[int]) -> dict[int, int]:
    series = pd.Series(list(labels), dtype=int)
    return {int(k): int(v) for k, v in series.value_counts().sort_index().items()}


def load_dataset_bundle(bundle_dir: str | Path) -> DatasetBundle:
    root = Path(bundle_dir)
    required = {
        "full_dataset": root / "full_dataset.csv",
        "original_test": root / "original_test.csv",
        "textfooler_test": root / "textfooler_test.csv",
        "trust_test": root / "trust_test.csv",
        "urgency_test": root / "urgency_test.csv",
        "emotion_test": root / "emotion_test.csv",
        "metadata": root / "metadata.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"实验数据包缺少文件: {missing}")

    return DatasetBundle(
        full_dataset=pd.read_csv(required["full_dataset"]),
        original_test=pd.read_csv(required["original_test"]),
        textfooler_test=pd.read_csv(required["textfooler_test"]),
        trust_test=pd.read_csv(required["trust_test"]),
        urgency_test=pd.read_csv(required["urgency_test"]),
        emotion_test=pd.read_csv(required["emotion_test"]),
        metadata=pd.read_json(required["metadata"], typ="series").to_dict(),
    )
