# -*- coding: utf-8 -*-
"""
信息抽取模块（最终版）

目标：
1) 统一与 UI：可显式传入 api_key；默认仍兼容环境变量。
2) JSON 严格模式优先；失败自动回退到普通文本→抽 JSON→校验。
3) 与 jsonschema 严格对齐，校验失败给出“路径级”可读错误。
4) 支持较“脏”的模型输出：```json 围栏、裸 JSON、带注释的前后缀等。
5) 对超长对话进行限长截断（字符级），保障稳定性与开销。

用法（例）：
    from extract_ie import extract_profile
    data = extract_profile(
        turns=asr_lines,                # list[str]，建议带 [T#] 行号
        schema=json.load(open("customer_profile.schema.json", "r", encoding="utf-8")),
        system_prompt=open("extract_profile_system.txt","r",encoding="utf-8").read(),
        model="gpt-5-mini",
        base_url=None,                  # 如你走自建网关则填
        api_key="sk-...",               # UI 传下来的 key；不传则走环境变量
        max_chars=6000,
        temperature=None,               # [MOD] 默认不传温度，避免 400
        seed=None,
        request_timeout=60.0,
    )

注意：
- schema 与提示词的“枚举定义/字段定义”必须一致，否则校验会失败。
- 如果你想保留“学生/上班族”等‘客户类型’，要么改 schema 的 enum，要么改提示词。
"""

import json
import re
from typing import List, Dict, Any, Optional

# [KEEP] OpenAI 官方 SDK
from openai import OpenAI
from openai._exceptions import OpenAIError

# [ADD] 更严格、可读的校验器
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError


# =========================
# 基础工具
# =========================
def _clip_turns(turns: List[str], max_chars: int) -> str:
    """[ADD] 将多行对话拼接，并在字符级做兜底截断，避免超长 prompt。"""
    joined = "\n".join(turns)
    if max_chars and len(joined) > max_chars:
        joined = joined[:max_chars] + "\n…(截断)"
    return joined


def _extract_outer_json(text: str) -> str:
    """[ADD] 从模型普通回答中尽量鲁棒地抽取最外层 JSON 文本。

    规则：
    1) 优先匹配 ```json ... ``` 围栏。
    2) 否则从第一个 '{' 到最后一个 '}' 的最大包围。
    3) 去掉可能的 BOM / 零宽字符。
    """
    # 优先 ```json 代码块
    fence = re.search(r"```json\s*(.+?)\s*```", text, flags=re.S | re.I)
    if fence:
        candidate = fence.group(1)
        candidate = candidate.strip()
        if candidate:
            return candidate

    # 退而求其次：最大花括号包围
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        candidate = candidate.strip()
        return candidate

    raise ValueError("未能在模型输出中找到可解析的 JSON。")


def _validate_with_readable_errors(data: Any, schema: Dict[str, Any]) -> None:
    """[ADD] 使用 Draft 2020-12 + FormatChecker，输出可读性更好的错误路径。"""
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        msgs = []
        for e in errors:
            path = "$" + "".join([f"/{str(p)}" for p in e.absolute_path])
            msgs.append(f"{path}: {e.message}")
        raise ValidationError("Schema 验证失败：\n" + "\n".join(msgs))


# =========================
# 主函数
# =========================
def extract_profile(
    turns: List[str],
    schema: Dict[str, Any],
    system_prompt: str,
    model: str,
    base_url: Optional[str] = None,
    *,
    max_chars: int = 6000,                 # [ADD] 本地兜底限长（字符）
    temperature: Optional[float] = None,   # [MOD] 温度可为 None（默认不发送）
    seed: Optional[int] = None,            # [ADD] 可重复性（若模型支持）
    request_timeout: float = 60.0,         # [ADD] 请求超时（秒）
    api_key: Optional[str] = None,         # [ADD] 与 UI 对齐：允许外部传入
) -> Dict[str, Any]:
    """
    根据 ASR 合成的对话文本（建议带行号）进行信息抽取，返回**已通过 schema 校验**的字典。
    """

    # [ADD] 组装 OpenAI 客户端（允许 base_url 和 api_key）
    client_kwargs: Dict[str, Any] = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key
    client = OpenAI(**client_kwargs)

    # [ADD] 限长后的对话输入
    text_block = _clip_turns(turns, max_chars=max_chars)

    # [ADD] 强约束：让模型**仅产出 JSON 对象**，不加解释文字
    sys_msg = (
        system_prompt.strip()
        + "\n\n【格式要求】你必须只输出一个 JSON 对象，不能包含任何额外说明或代码块围栏。\n"
        + "【注意】所有字段必须符合给定的 JSON Schema；若无法判断的字段，填入合理的默认值或 'unknown'。"
    )

    messages = [
        {"role": "system", "content": sys_msg},
        {
            "role": "user",
            "content": (
                "以下是带行号的对话文本，请抽取信息并返回严格符合 Schema 的 JSON：\n\n"
                + text_block
            ),
        },
    ]

    # [ADD] 优先：JSON-mode
    raw = None  # [ADD] 兜底，防止异常路径下未定义
    try:
        # JSON 模式：不传 temperature（避免部分模型报 400）
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=seed,
            timeout=request_timeout,
            response_format={"type": "json_object"},  # 强制 JSON
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)  # 直接反序列化
    except (OpenAIError, json.JSONDecodeError, KeyError, IndexError) as e_json_mode:
        # [ADD] 回退：不使用 response_format，改走普通文本→抽 JSON
        try:
            # [MOD] 仅当 temperature 不为 None 时才传入，避免 400
            _common = dict(model=model, messages=messages, seed=seed, timeout=request_timeout)
            if temperature is not None:
                _common["temperature"] = temperature

            resp2 = client.chat.completions.create(**_common)
            raw = (resp2.choices[0].message.content or "").strip()
            extracted = _extract_outer_json(raw)
            data = json.loads(extracted)
        except Exception as e_fallback:
            # [ADD] 统一抛出可定位问题的错误，包含预览片段
            preview = (raw or "")[:400] + (" ..." if raw and len(raw) > 400 else "")
            raise RuntimeError(
                "信息抽取失败（JSON-mode 与回退模式均失败）。\n"
                f"- model: {model}\n"
                f"- base_url: {base_url or '默认'}\n"
                f"- json-mode error: {repr(e_json_mode)}\n"
                f"- fallback error: {repr(e_fallback)}\n"
                f"- raw preview: {preview}"
            ) from e_fallback

    # [ADD] 严格 schema 校验（可读报错路径）
    try:
        _validate_with_readable_errors(data, schema)
    except ValidationError as ve:
        raw_preview = json.dumps(data, ensure_ascii=False)[:400] + " ..."
        raise ValidationError(
            "模型返回 JSON 与 schema 不一致。\n"
            f"- model: {model}\n"
            f"- base_url: {base_url or '默认'}\n"
            f"- detail: {ve}\n"
            f"- json preview: {raw_preview}"
        ) from ve

    return data
