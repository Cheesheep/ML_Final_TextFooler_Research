from __future__ import annotations

from typing import Callable


def _replace_first_left(text: str, injected: str) -> str:
    marker = "left:"
    if marker not in text:
        return injected + "\n" + text
    return text.replace(marker, f"{marker} {injected} ", 1)


def _replace_second_left(text: str, injected: str) -> str:
    marker = "left:"
    first = text.find(marker)
    if first == -1:
        return _replace_first_left(text, injected)
    second = text.find(marker, first + len(marker))
    if second == -1:
        return _replace_first_left(text, injected)
    return text[: second + len(marker)] + f" {injected} " + text[second + len(marker) :]


def _replace_many(text: str, replacements: list[tuple[str, str]]) -> str:
    updated = text
    for old, new in replacements:
        updated = updated.replace(old, new)
    return updated


def trust_building_rewrite(text: str) -> str:
    injected = "您好，我这边是平台官方售后中心，工号A3172，之前已经为您登记过这笔业务。"
    updated = _replace_first_left(text, injected)
    return _replace_many(
        updated,
        [
            ("点击一个链接", "进入官方核验页面"),
            ("点击我发送给你的链接", "打开我给您发送的服务页面"),
            ("下载一个官方的应用程序", "通过官方服务页面完成处理"),
            ("下载我们专门的客户服务APP", "通过我们的线上客服通道完成处理"),
            ("银行卡信息", "收款账户信息"),
            ("银行卡短信", "收款账户信息"),
            ("微信", "客服通道"),
        ],
    )


def urgency_rewrite(text: str) -> str:
    injected = "系统显示这个处理通道只保留三十分钟，如果超时就会自动关闭，麻烦您尽快操作。"
    updated = _replace_second_left(text, injected)
    return _replace_many(
        updated,
        [
            ("退款", "异常处理"),
            ("下载", "打开"),
            ("链接", "处理页面"),
            ("点击", "进入"),
            ("APP", "服务页面"),
            ("验证", "核验"),
        ],
    )


def emotion_rewrite(text: str) -> str:
    injected = "真的不好意思耽误您时间，我也是想尽快帮您把这件事处理好，不然我这边也没法提交。"
    updated = _replace_second_left(text, injected)
    return _replace_many(
        updated,
        [
            ("银行卡信息", "账户信息"),
            ("点击", "打开"),
            ("链接", "页面"),
            ("下载", "安装"),
            ("退款", "处理"),
            ("验证", "确认"),
        ],
    )


REWRITE_FUNCTIONS: dict[str, Callable[[str], str]] = {
    "trust": trust_building_rewrite,
    "urgency": urgency_rewrite,
    "emotion": emotion_rewrite,
}
