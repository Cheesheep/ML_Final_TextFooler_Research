from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fraud_call_benchmark.data import audit_attack_pairs, load_attack_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 TextFooler 攻击样本文件")
    parser.add_argument("--attack-xlsx", required=True, help="攻击样本 Excel 路径")
    parser.add_argument(
        "--output-json",
        default="outputs/attack_audit.json",
        help="审计结果输出路径",
    )
    args = parser.parse_args()

    # 这里先把原始 Excel 统一读成标准字段，后面的审计逻辑就只关心文本本身了。
    frame = load_attack_pairs(args.attack_xlsx)
    audit = audit_attack_pairs(frame)

    payload = {
        "rows": audit.rows,
        "columns": audit.columns,
        "avg_original_chars": round(audit.avg_original_chars, 3),
        "avg_attacked_chars": round(audit.avg_attacked_chars, 3),
        "avg_original_turns": round(audit.avg_original_turns, 3),
        "avg_attacked_turns": round(audit.avg_attacked_turns, 3),
        "preview": frame.head(3).to_dict(orient="records"),
    }

    # 审计结果单独落一个 json，后面写实验说明和核对数据时都会直接用到。
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
