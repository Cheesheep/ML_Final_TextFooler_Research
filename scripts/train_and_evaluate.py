from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_call_benchmark.data import load_dataset_bundle
from fraud_call_benchmark.metrics import classification_metrics


def evaluate_frame(model: Pipeline, frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    # 指标和逐条预测结果一起回传，后面导表和查误判都靠这一份结果。
    preds = model.predict(frame["text"].tolist())
    result_frame = frame.copy()
    result_frame["prediction"] = preds
    result_frame["correct"] = (result_frame["prediction"] == result_frame["label"]).astype(int)
    return classification_metrics(frame["label"].to_numpy(), preds), result_frame


def build_model() -> Pipeline:
    # 这里保持成一条标准 sklearn pipeline，训练和推理时走的是同一套预处理。
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def build_metrics_table(results: dict) -> pd.DataFrame:
    # json 结果更适合机器读，表格更适合论文和人工核对，两份都保留。
    rows = []
    for name, metrics in results["evaluations"].items():
        rows.append(
            {
                "evaluation_set": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )
    return pd.DataFrame(rows)


def plot_metric_bars(metrics_table: pd.DataFrame, output_dir: Path) -> None:
    # 图只画几项核心指标，方便直接看不同改写测试集之间的落差。
    plt.figure(figsize=(11, 6))
    melted = metrics_table.melt(
        id_vars=["evaluation_set"],
        value_vars=["accuracy", "precision", "recall", "f1"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=melted, x="evaluation_set", y="value", hue="metric")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Score")
    plt.xlabel("")
    plt.title("Performance Comparison Across Original and Rewritten Test Sets")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison.png", dpi=200)
    plt.close()


def build_delta_table(metrics_table: pd.DataFrame) -> pd.DataFrame:
    # 所有改写结果都拿 original_test 当参照，这样对比口径是一致的。
    base = metrics_table[metrics_table["evaluation_set"] == "original_test"].iloc[0]
    rows = []
    for _, row in metrics_table.iterrows():
        rows.append(
            {
                "evaluation_set": row["evaluation_set"],
                "accuracy_delta": round(row["accuracy"] - base["accuracy"], 6),
                "precision_delta": round(row["precision"] - base["precision"], 6),
                "recall_delta": round(row["recall"] - base["recall"], 6),
                "f1_delta": round(row["f1"] - base["f1"], 6),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练并评测虚假通话分类模型")
    parser.add_argument(
        "--bundle-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "fraud_benchmark"),
        help="prepare_experiment_data.py 生成的数据包目录",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "final"))
    args = parser.parse_args()

    # bundle 里已经把 full_dataset 和几类测试变体都整理好了，训练脚本只负责消费。
    bundle = load_dataset_bundle(args.bundle_dir)
    full = bundle.full_dataset.copy()
    train = full[full["split"] == "train"].copy()
    val = full[full["split"] == "val"].copy()

    model = build_model()
    model.fit(train["text"].tolist(), train["label"].tolist())

    frames = {
        "val": val,
        "original_test": bundle.original_test,
        "textfooler_test": bundle.textfooler_test,
        "trust_test": bundle.trust_test,
        "urgency_test": bundle.urgency_test,
        "emotion_test": bundle.emotion_test,
    }
    evaluations: dict[str, dict[str, float]] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    # 同一个模型顺次跑各个测试版本，保证对比只来自文本改写，不来自模型波动。
    for name, frame in frames.items():
        metrics, pred_frame = evaluate_frame(model, frame)
        evaluations[name] = metrics
        prediction_frames[name] = pred_frame

    metrics_table = build_metrics_table({"evaluations": evaluations})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(output_dir / "metrics_table.csv", index=False, encoding="utf-8-sig")
    delta_table = build_delta_table(metrics_table)
    delta_table.to_csv(output_dir / "metrics_delta_vs_original.csv", index=False, encoding="utf-8-sig")

    details_dir = output_dir / "prediction_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    for name, pred_frame in prediction_frames.items():
        pred_frame.to_csv(details_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    # 漏判正类单独导出来，后面做案例分析不用再翻整张预测表。
    original_fn = prediction_frames["original_test"]
    original_fn = original_fn[(original_fn["label"] == 1) & (original_fn["prediction"] == 0)].copy()
    if not original_fn.empty:
        original_fn.to_csv(output_dir / "false_negative_cases_original.csv", index=False, encoding="utf-8-sig")

    results = {
        "dataset_metadata": bundle.metadata,
        "model": {
            "type": "TF-IDF(char 2-4 gram) + LogisticRegression(liblinear)",
        },
        "evaluations": evaluations,
    }
    (output_dir / "experiment_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_metric_bars(metrics_table, output_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
