# -*- coding: utf-8 -*-
"""
extract_ie.py — 信息抽取（健壮版）
改动要点：
- [ADD] 在 system 指令中**内嵌 schema 文本**，让模型严格对齐字段与类型
- [ADD] 先尝试 JSON-mode；若失败，**回退普通文本**并做外层 JSON 截取
- [ADD] 更可读的 jsonschema 校验：Draft2020 + format_checker + 聚合报错路径
- [ADD] 限长兜底（字符级），避免极端长输入直接超限
- [ADD] 详细异常上下文：模型名 / base_url / 片段预览
- [ADD] 可选 max_chars, temperature, seed 等参数
- [ADD] 明确超时（OpenAI SDK）
"""

import json
import re
from typing import List, Dict, Any, Optional

from openai import OpenAI
from openai._exceptions import OpenAIError

# [ADD] 更严格、可读的校验器
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError


def _truncate_turns(turns: List[str], max_chars: int) -> List[str]:
    """[ADD] 简单字符限长兜底，避免提示超长；上层也可再限长"""
    buf, total = [], 0
    for t in turns:
        if total + len(t) <= max_chars:
            buf.append(t)
            total += len(t)
        else:
            remain = max_chars - total
            if remain > 0:
                buf.append(t[:remain])
            break
    return buf


def _extract_outer_json(text: str) -> str:
    """[ADD] 从一段可能包含说明/围栏的文本里，抓最外层 JSON"""
    # 去除 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fence:
        return fence.group(1)

    l, r = text.find("{"), text.rfind("}")
    if 0 <= l < r:
        return text[l : r + 1]
    return text  # 兜底返回原文（后续 json.loads 会抛错）


def _validate_with_readable_errors(instance: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """[ADD] 使用 Draft2020-12 + format 校验，并给出可读的错误路径"""
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if errors:
        msgs = []
        for e in errors:
            path = "$" + "".join(f"[{repr(p)}]" if isinstance(p, int) else f".{p}" for p in e.path)
            msgs.append(f"{path}: {e.message}")
        raise ValidationError("Schema 验证失败：\n" + "\n".join(msgs))


def extract_profile(
    turns: List[str],
    schema: Dict[str, Any],
    system_prompt: str,
    model: str,
    base_url: Optional[str] = None,
    *,
    max_chars: int = 6000,         # [ADD] 本地兜底限长（字符）
    temperature: float = 0.1,      # [ADD] 可调
    seed: Optional[int] = None,    # [ADD] 可重复性（若模型支持）
    request_timeout: float = 60.0, # [ADD] 超时（秒）
) -> Dict[str, Any]:
    """
    返回：经严格 schema 校验后的 dict
    失败会抛出带上下文的异常（包含模型名、base_url、返回片段）
    """
    # [ADD] 兜底限长（上层已有也不冲突）
    safe_turns = _truncate_turns(turns or [], max_chars=max_chars)
    user_text = "\n".join(safe_turns)

    # [ADD] 把 schema 文本注入到 system 指令，让模型“看得见”字段与类型
    #       注意：若 schema 很大，可考虑只给出 properties 的关键字段与类型示例
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

    sys_msg = (
        system_prompt.strip() + "\n\n"
        "你必须严格输出一个 JSON 对象，且字段与类型满足以下 JSON Schema（Draft 2020-12 语义）：\n"
        f"{schema_str}\n"
        "不要输出任何解释、前后缀或 Markdown，只输出 JSON 对象本体。"
    )

    client_kwargs = {"timeout": request_timeout}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    # [ADD] 首选 JSON-mode；失败回退普通文本 + 手动解析
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            seed=seed,
            response_format={"type": "json_object"},  # 仅在部分模型/网关生效
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": "以下是带行号的转写文本，请按 schema 输出 JSON：\n" + user_text},
            ],
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)  # 若网关不支持 JSON-mode，可能抛错
    except Exception as e_json_mode:
        # [ADD] 回退路径：不带 JSON-mode，再手动抽 JSON
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                seed=seed,
                messages=[
                    {"role": "system", "content": sys_msg + "\n（注意：只输出 JSON 对象，不要其他文字。）"},
                    {"role": "user", "content": "以下是带行号的转写文本，请按 schema 输出 JSON：\n" + user_text},
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = _extract_outer_json(raw)
            data = json.loads(raw)
        except Exception as e_fallback:
            # [ADD] 给出更清晰的上下文，便于排障
            raw_preview = (raw[:400] + " ...") if isinstance(raw, str) and len(raw) > 400 else raw
            raise RuntimeError(
                "LLM 返回无法解析为 JSON；JSON-mode 与普通模式均失败。\n"
                f"- model: {model}\n"
                f"- base_url: {base_url or '默认'}\n"
                f"- error(JSON-mode): {type(e_json_mode).__name__}: {e_json_mode}\n"
                f"- error(fallback): {type(e_fallback).__name__}: {e_fallback}\n"
                f"- raw preview: {raw_preview}"
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
