"""
信息抽取模块（容错增强版：不报错、自动修剪与补齐）

目标：
1) 统一与 UI：可显式传入 api_key；默认仍兼容环境变量。
2) JSON 严格模式优先；失败自动回退到普通文本→抽 JSON。
3) 对 LLM 返回进行“自动修剪/补齐/清洗”，再做 schema 校验；若仍失败，回退到最小合规骨架。
4) 绝不 raise —— 始终返回一个“与 schema 完全一致”的 JSON；问题写入 warnings。

与 schema / prompt 的匹配：
- schema：你上传的 customer_profile.schema.json（不包含 lifestyle.value），additionalProperties=false。
- prompt：你上传的 extract_profile_system_json.txt（明确禁止多余字段）。
二者与本文件逻辑完全匹配。
"""

import json
import re
from copy import deepcopy
from typing import List, Dict, Any, Optional

# OpenAI SDK
from openai import OpenAI
from openai._exceptions import OpenAIError

# jsonschema 校验
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError


# =========================
# 基础工具
# =========================
def _clip_turns(turns: List[str], max_chars: int) -> str:
    """将多行对话拼接，并做字符级兜底截断。"""
    joined = "\n".join(turns)
    if max_chars and len(joined) > max_chars:
        joined = joined[:max_chars] + "\n…(截断)"
    return joined


def _extract_outer_json(text: str) -> str:
    """从普通回答中鲁棒抽取最外层 JSON 文本。"""
    fence = re.search(r"```json\s*(.+?)\s*```", text, flags=re.S | re.I)
    if fence:
        candidate = fence.group(1).strip()
        if candidate:
            return candidate
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    raise ValueError("未能在模型输出中找到可解析的 JSON。")


def _validator(schema: Dict[str, Any]) -> Draft202012Validator:
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _validate(data: Any, schema: Dict[str, Any]) -> None:
    """严格校验，失败抛 ValidationError。"""
    v = _validator(schema)
    errs = sorted(v.iter_errors(data), key=lambda e: e.path)
    if errs:
        msgs = []
        for e in errs:
            path = "$" + "".join([f"/{str(p)}" for p in e.absolute_path])
            msgs.append(f"{path}: {e.message}")
        raise ValidationError("Schema 验证失败：\n" + "\n".join(msgs))


# =========================
# 容错修复工具（核心）
# =========================
def _clean_confidence(x) -> float:
    try:
        val = float(x)
    except Exception:
        return 0.0
    return 0.0 if val < 0 else (1.0 if val > 1 else val)


def _clean_evidence(arr, max_turns: Optional[int] = None) -> List[int]:
    if not isinstance(arr, list):
        return []
    out = []
    for v in arr:
        if isinstance(v, int) and v > 0:
            if (max_turns is None) or (v <= max_turns):
                out.append(v)
    return sorted(set(out))


def _enum_or_unknown(val: Any, enum: List[str]) -> str:
    if isinstance(val, str) and val in enum:
        return val
    return "unknown" if "unknown" in enum else enum[0]


def _type_default(t: str) -> Any:
    # 当需要自动补齐必填字段时的默认值
    return {"string": "unknown", "number": 0.0, "integer": 0, "array": [], "object": {}, "boolean": False}.get(t, None)


def _prune_and_fill(data: Any, schema: Dict[str, Any], warnings: List[str]) -> Any:
    """
    递归：
    - 删除未声明键（additionalProperties=false）
    - 补齐 required 字段（用 default 或类型默认值/unknown）
    - 对 enum 键做容错为 'unknown'
    - 对 confidence / evidence_turns 等做清洗
    """
    def walk(x, sch):
        t = sch.get("type")
        if t == "object" and isinstance(x, dict):
            props = sch.get("properties", {})
            ap = sch.get("additionalProperties", True)
            # 先删未声明键
            if ap is False:
                for k in list(x.keys()):
                    if k not in props:
                        x.pop(k, None)
                        warnings.append(f"已移除未声明字段: {k}")
            # 递归到子字段
            for k, sub in props.items():
                if k in x:
                    x[k] = walk(x[k], sub)
            # 补齐 required
            for rk in sch.get("required", []):
                if rk not in x:
                    dv = props.get(rk, {}).get("default", None)
                    if dv is None:
                        dv = _type_default(props.get(rk, {}).get("type", "string"))
                    x[rk] = dv
                    warnings.append(f"已补齐必填字段: {rk} -> {dv}")
            return x

        if t == "array" and isinstance(x, list):
            items_schema = sch.get("items", {})
            return [walk(i, items_schema) for i in x]

        # 叶子节点：做枚举/数值清洗
        if "enum" in sch:
            return _enum_or_unknown(x, sch["enum"])

        # 针对特定命名的小清洗（非强依赖 schema）
        return x

    # 先深拷贝，避免副作用
    y = deepcopy(data)
    y = walk(y, schema)

    # 针对你的结构再做定制清洗
    cp = y.get("customer_profile", {})
    # customer_type / lifestyle / family_status / income 四块的通用清洗
    for key in ("customer_type", "lifestyle", "family_status", "income"):
        blk = cp.get(key)
        if isinstance(blk, dict):
            # confidence 0~1
            if "confidence" in blk:
                old = blk["confidence"]
                blk["confidence"] = _clean_confidence(old)
                if blk["confidence"] != old:
                    warnings.append(f"{key}.confidence 已规范化到 [0,1]")
            # evidence_turns 列表清洗
            if "evidence_turns" in blk:
                old = blk["evidence_turns"]
                blk["evidence_turns"] = _clean_evidence(old)
                if blk["evidence_turns"] != old:
                    warnings.append(f"{key}.evidence_turns 已清洗为正整数去重列表")
            # lifestyle 里若模型放了 value/summary，挪到 consumption_habits（仅当后者 unknown/空）
            if key == "lifestyle":
                summary = None
                if "value" in blk:
                    summary = blk.get("value")
                    blk.pop("value", None)
                    warnings.append("lifestyle.value 已移除")
                if "summary" in blk and not summary:
                    summary = blk.get("summary")
                    blk.pop("summary", None)
                    warnings.append("lifestyle.summary 已移除")
                if summary:
                    ch = (blk.get("consumption_habits") or "").strip().lower()
                    if (not ch) or ch == "unknown":
                        blk["consumption_habits"] = summary
                        warnings.append("lifestyle.value/summary 已转存到 consumption_habits")
    return y


def _minimal_skeleton(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成一个“最小合规骨架”，保证能通过校验。
    - 填充 required（递归），值使用 default 或类型默认值。
    - 增加 warnings 提示“骨架回退”。
    """
    def build(sch):
        t = sch.get("type")
        if t == "object":
            obj = {}
            props = sch.get("properties", {})
            for rk in sch.get("required", []):
                sub = props.get(rk, {})
                dv = sub.get("default", None)
                if dv is None:
                    dv = _type_default(sub.get("type", "string"))
                obj[rk] = build(sub) if sub.get("type") in ("object", "array") else dv
            # 非 required 的先不加，保持最小
            return obj
        if t == "array":
            return []
        if "enum" in sch:
            return _enum_or_unknown(None, sch["enum"])
        dv = sch.get("default", None)
        if dv is not None:
            return dv
        return _type_default(t or "string")

    base = build(schema)
    # 顶层附加 warnings（若定义了该字段）
    if isinstance(base, dict) and "warnings" in schema.get("properties", {}):
        base["warnings"] = ["回退到最小合规骨架（原始结果无法自动修复）"]
    return base


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
    max_chars: int = 6000,
    temperature: Optional[float] = None,  # 默认不传温度，避免个别模型 400
    seed: Optional[int] = None,
    request_timeout: float = 60.0,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    返回：始终为“通过 schema 校验”的 JSON。
    - 若 LLM 输出有瑕疵，将自动修剪/补齐/清洗；不可修复则回退到最小骨架，并在 warnings 里说明。
    """

    warnings: List[str] = []

    # OpenAI client
    client_kwargs: Dict[str, Any] = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key
    client = OpenAI(**client_kwargs)

    # 输入拼接
    text_block = _clip_turns(turns, max_chars=max_chars)

    # System 提示（强调只输出 JSON）
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

    # Step 1: JSON-mode
    raw = None
    data = None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=seed,
            timeout=request_timeout,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw)
    except Exception as e_json_mode:
        # Step 1 fallback: 普通回答→抽 JSON
        try:
            common = dict(model=model, messages=messages, seed=seed, timeout=request_timeout)
            if temperature is not None:
                common["temperature"] = temperature
            resp2 = client.chat.completions.create(**common)
            raw = (resp2.choices[0].message.content or "").strip()
            extracted = _extract_outer_json(raw)
            data = json.loads(extracted)
        except Exception as e_fallback:
            # LLM 完全失败：直接回退骨架
            skel = _minimal_skeleton(schema)
            if isinstance(skel, dict):
                skel.setdefault("warnings", []).append(
                    "LLM 输出解析失败，已回退到最小合规骨架。"
                )
            return skel

    # Step 2: 自动修剪/补齐/清洗 → 校验
    try:
        cleaned = _prune_and_fill(data, schema, warnings)
        # 第一次校验
        _validate(cleaned, schema)
        # 附加 warnings（若 schema 顶层允许）
        if isinstance(cleaned, dict) and "warnings" in schema.get("properties", {}):
            cleaned.setdefault("warnings", [])
            cleaned["warnings"].extend(warnings)
        return cleaned
    except ValidationError as ve:
        # Step 3: 一次“自我修复”重试（把错误发回模型要求修正）
        try:
            err_msg = str(ve)
            fix_messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content":
                    "你上次返回的 JSON 未通过校验，错误如下：\n"
                    + err_msg
                    + "\n\n请严格按 schema 修正并仅输出 JSON 对象（不要解释、不要代码围栏）。\n"
                    + text_block
                },
            ]
            resp_fix = client.chat.completions.create(
                model=model, messages=fix_messages,
                seed=seed, timeout=request_timeout,
                response_format={"type": "json_object"},
            )
            raw_fix = (resp_fix.choices[0].message.content or "").strip()
            data_fix = json.loads(raw_fix)
            cleaned_fix = _prune_and_fill(data_fix, schema, warnings)
            _validate(cleaned_fix, schema)
            if isinstance(cleaned_fix, dict) and "warnings" in schema.get("properties", {}):
                cleaned_fix.setdefault("warnings", [])
                cleaned_fix["warnings"].extend(warnings + ["已通过一次自我修复重试"])
            return cleaned_fix
        except Exception:
            # Step 4: 依旧不合规 → 直接回退最小骨架，并把可读问题写进 warnings
            skel = _minimal_skeleton(schema)
            if isinstance(skel, dict):
                skel.setdefault("warnings", [])
                skel["warnings"].extend(warnings)
                skel["warnings"].append("模型返回的 JSON 与 schema 不一致，已回退到最小合规骨架。")
            return skel
