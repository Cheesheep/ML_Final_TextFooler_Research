# 实验结果摘要

## 数据构成

- 欺诈正类：2677
- 正常负类：2677
- 总样本：5354
- 划分：
  - train: 3747
  - val: 536
  - test: 1071

## 模型

- TF-IDF(char 2-4 gram) + LogisticRegression(liblinear)

## 主要结果

| 测试集 | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| 原始测试集 | 0.997199 | 1.000000 | 0.994393 | 0.997188 |
| TextFooler 攻击测试集 | 0.997199 | 1.000000 | 0.994393 | 0.997188 |
| 信任建立改写测试集 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 紧迫感强化改写测试集 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 情感操纵改写测试集 | 0.997199 | 1.000000 | 0.994393 | 0.997188 |

## 结论

1. TextFooler 在当前实验设定下没有造成明显性能下降。
2. 信任建立和紧迫感改写使 Recall 从 0.994393 提升到 1.000000。
3. 误判样本集中在绑架勒索类诈骗，而不是客服退款类诈骗。

## 关键文件

- `experiment_results.json`
- `metrics_table.csv`
- `metrics_delta_vs_original.csv`
- `false_negative_cases_original.csv`
- `performance_comparison.png`
