# 虚假通话检测实验复现子集

该目录仅保留实验复现相关内容，不包含论文正文、课程说明、过程文档和其他非实验交付文件。

## 保留内容

- 课程原始攻击数据：`data/TextFooler攻击后新的欺诈通话数据.xlsx`
- 已处理实验数据包：`data/processed/fraud_benchmark/`
- 训练与评测脚本：`scripts/`
- 实验源码：`src/fraud_call_benchmark/`
- 最终实验输出：`outputs/final/`

## 直接可运行的命令

```powershell
python -m pip install -r requirements.txt
python scripts\audit_dataset.py --attack-xlsx "data\TextFooler攻击后新的欺诈通话数据.xlsx"
python scripts\train_and_evaluate.py --bundle-dir "data\processed\fraud_benchmark" --output-dir "outputs\final"
```

## 说明

- 该导出目录已经包含处理后的实验数据与最终结果文件，适合做实验展示和结果复核。
- `scripts/prepare_experiment_data.py` 被一并保留，但其完整重建流程依赖未导出的 `data/raw/` 原始 LCCC 压缩文件。

