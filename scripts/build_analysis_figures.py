from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FONT = Path(r"C:\Windows\Fonts\msyh.ttc")
TITLE_FONT = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    font_path = TITLE_FONT if bold and TITLE_FONT.exists() else DEFAULT_FONT
    return ImageFont.truetype(str(font_path), size=size)


def wrap_text(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def draw_rounded_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str) -> None:
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=3)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str) -> None:
    draw.line([start, end], fill=color, width=5)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) >= abs(dy):
        if dx >= 0:
            head = [(end[0], end[1]), (end[0] - 18, end[1] - 10), (end[0] - 18, end[1] + 10)]
        else:
            head = [(end[0], end[1]), (end[0] + 18, end[1] - 10), (end[0] + 18, end[1] + 10)]
    else:
        if dy >= 0:
            head = [(end[0], end[1]), (end[0] - 10, end[1] - 18), (end[0] + 10, end[1] - 18)]
        else:
            head = [(end[0], end[1]), (end[0] - 10, end[1] + 18), (end[0] + 10, end[1] + 18)]
    draw.polygon(head, fill=color)


def centered_multiline(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font, fill: str) -> None:
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=8, align="center")
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + (right - left - text_w) / 2
    y = top + (bottom - top - text_h) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=8, align="center")


def build_flowchart(output_path: Path) -> None:
    image = Image.new("RGB", (1800, 1100), "#fbfcfd")
    draw = ImageDraw.Draw(image)
    title_font = load_font(44, bold=True)
    text_font = load_font(28)
    foot_font = load_font(24)

    draw.text((420, 50), "虚假通话检测项目研究流程图", font=title_font, fill="#18344a")
    boxes = [
        ((90, 220, 500, 430), "#f9d9dc", "课程欺诈通话原文\n2677 条正类"),
        ((90, 560, 500, 770), "#dceffd", "LCCC 正常对话筛选\n2677 条负类"),
        ((690, 390, 1110, 600), "#fcf0c9", "平衡二分类数据集\n共 5354 条"),
        ((1290, 220, 1700, 430), "#e9def5", "分层划分\ntrain / val / test"),
        ((1290, 560, 1700, 770), "#d9f3e6", "字符级 TF-IDF +\nLogistic Regression"),
        ((640, 830, 1160, 1010), "#f8e7a3", "鲁棒性评测\n原始 / TextFooler / trust / urgency / emotion"),
    ]
    for box, fill, text in boxes:
        draw_rounded_box(draw, box, fill, "#35516b")
        centered_multiline(draw, box, text, text_font, "#1f1f1f")

    arrows = [
        ((500, 325), (690, 480)),
        ((500, 665), (690, 510)),
        ((1110, 495), (1290, 325)),
        ((1495, 430), (1495, 560)),
        ((1290, 665), (1160, 915)),
        ((900, 600), (900, 830)),
    ]
    for start, end in arrows:
        draw_arrow(draw, start, end, "#35516b")

    note = "课程数据仅提供欺诈正类及其 TextFooler 版本，因此实验中补充 LCCC 负类以完成监督训练。"
    draw.text((270, 1040), note, font=foot_font, fill="#5b6770")
    image.save(output_path)


def normalize_dialogue(text: str) -> str:
    return text.replace("left:", "诈骗方：").replace("right:", "受害者：").replace("\r", "").strip()


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def build_false_negative_case_figure(
    false_negative_path: Path,
    trust_path: Path,
    urgency_path: Path,
    output_path: Path,
) -> None:
    false_negative = read_cases(false_negative_path)
    trust = {row["pair_id"]: row for row in read_cases(trust_path)}
    urgency = {row["pair_id"]: row for row in read_cases(urgency_path)}

    image = Image.new("RGB", (2200, 1700), "#fbfcfd")
    draw = ImageDraw.Draw(image)
    title_font = load_font(42, bold=True)
    label_font = load_font(24, bold=True)
    body_font = load_font(21)
    note_font = load_font(20)

    draw.text((510, 40), "误判样本在 trust / urgency 改写后的预测修正情况", font=title_font, fill="#18344a")

    y_positions = [180, 690, 1200]
    for idx, row in enumerate(false_negative):
        pair_id = row["pair_id"]
        raw_text = normalize_dialogue(row[row.keys().__iter__().__next__()])
        trust_text = normalize_dialogue(trust[pair_id][next(iter(trust[pair_id].keys()))])
        urgency_text = normalize_dialogue(urgency[pair_id][next(iter(urgency[pair_id].keys()))])
        y = y_positions[idx]

        draw.text((70, y - 45), f"pair_id = {pair_id} | 原始预测 0，改写后预测 1", font=label_font, fill="#2c3e50")

        left_box = (70, y, 780, y + 340)
        right_top_box = (1030, y, 2130, y + 155)
        right_bottom_box = (1030, y + 185, 2130, y + 340)

        draw_rounded_box(draw, left_box, "#fdecea", "#c0392b")
        draw_rounded_box(draw, right_top_box, "#eafaf1", "#27ae60")
        draw_rounded_box(draw, right_bottom_box, "#fef5e7", "#e67e22")

        draw.text((105, y + 18), wrap_text(raw_text, 22), font=body_font, fill="#2b2b2b", spacing=8)
        draw.text((1060, y + 14), "trust 改写", font=label_font, fill="#1e8449")
        draw.text((1060, y + 54), wrap_text(trust_text, 40), font=body_font, fill="#2b2b2b", spacing=8)
        draw.text((1060, y + 198), "urgency 改写", font=label_font, fill="#af601a")
        draw.text((1060, y + 238), wrap_text(urgency_text, 40), font=body_font, fill="#2b2b2b", spacing=8)

        draw_arrow(draw, (820, y + 155), (980, y + 155), "#35516b")
        draw.text((840, y + 105), "预测 0 -> 1", font=label_font, fill="#35516b")

    note = "左侧为原始测试集中漏判的绑架勒索类样本，右侧分别为 trust 与 urgency 改写版本。"
    draw.text((70, 1625), note, font=note_font, fill="#5b6770")
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成论文补充图片")
    parser.add_argument(
        "--result-dir",
        default=str(PROJECT_ROOT / "outputs" / "final"),
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    build_flowchart(result_dir / "research_flowchart.png")
    build_false_negative_case_figure(
        result_dir / "false_negative_cases_original.csv",
        result_dir / "prediction_details" / "trust_test.csv",
        result_dir / "prediction_details" / "urgency_test.csv",
        result_dir / "false_negative_comparison.png",
    )


if __name__ == "__main__":
    main()
