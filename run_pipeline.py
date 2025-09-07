"""
run_pipeline.py  — 完全替换版
特性：
1) 分阶段执行器：任何异常都会定位到“失败阶段：XXX”
2) 双日志：文件(详尽 DEBUG) + 控制台(简要 INFO)；UI 可查看最近 N 行
3) 显式自检：OpenAI Key / Base URL / Schema / Prompt
4) ASR 结果为空的硬校验；LLM JSON 返回体的类型校验
5) 仅使用相对路径；目录自动创建
6) 支持 Gradio GUI 与 CLI 两种入口

依赖（建议）：
- gradio>=4
- openai>=1.0.0
- faster-whisper>=1.0.0（在 asr.py 内部用）
"""

import os
import sys
import time
import json
import traceback
import argparse
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from contextlib import contextmanager
from typing import List, Dict, Any

# ---------------------------
# 兼容导入（包式/扁平式目录都可）
# ---------------------------
try:
    from configs import config as _config_mod  # 如果你有包式结构
    from pipelines.asr import transcribe as _asr_transcribe
    CONFIG_IMPORTED_FROM = "package"
except Exception:
    import importlib
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
    _config_mod = importlib.import_module("config")
    _asr_transcribe = importlib.import_module("asr").transcribe
    CONFIG_IMPORTED_FROM = "flat"

config = _config_mod  # 统一命名

# ---------------------------
# 日志环 & 常量
# ---------------------------
_UI_RING = deque(maxlen=getattr(config, "MAX_LOG_LINES_SHOWN", 200))

# ---------------------------
# 日志初始化
# ---------------------------
def _init_logger():
    os.makedirs(getattr(config, "LOG_DIR", "outputs/logs"), exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    trace_id = f"run_{ts}"
    log_path = os.path.join(getattr(config, "LOG_DIR", "outputs/logs"), f"{trace_id}.log")

    logger = logging.getLogger(trace_id)
    logger.setLevel(logging.DEBUG)

    fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger, trace_id, log_path


def _ui_log(logger, level, msg):
    level = level.lower()
    getattr(logger, level)(msg)
    _UI_RING.append(msg)


@contextmanager
def _stage(logger, name):
    t0 = time.time()
    _ui_log(logger, "INFO", f"▶ 开始阶段：{name}")
    try:
        yield
        dt = (time.time() - t0) * 1000.0
        _ui_log(logger, "INFO", f"✓ 完成阶段：{name}（{dt:.1f} ms）")
    except Exception as e:
        _ui_log(logger, "ERROR", f"✗ 阶段失败：{name} | {e.__class__.__name__}: {e}")
        _ui_log(logger, "ERROR", traceback.format_exc())
        raise


# ---------------------------
# 基础工具
# ---------------------------
def _ensure_dirs():
    os.makedirs(getattr(config, "AUDIO_DIR", "data/raw"), exist_ok=True)
    os.makedirs(getattr(config, "OUTPUT_DIR", "outputs/reports"), exist_ok=True)
    os.makedirs(getattr(config, "LOG_DIR", "outputs/logs"), exist_ok=True)


def _prepare_audio_file(source_mode: str, uploaded_file_path: str, manual_path: str) -> str:
    """
    返回可读音频文件的绝对路径（不做复制，直接使用给定路径/上传临时路径）。
    """
    if source_mode == "upload":
        if not uploaded_file_path:
            raise FileNotFoundError("未检测到上传音频。")
        if not os.path.exists(uploaded_file_path):
            raise FileNotFoundError(f"上传音频路径不存在：{uploaded_file_path}")
        return uploaded_file_path
    elif source_mode == "manual":
        p = (manual_path or "").strip()
        if not p:
            raise FileNotFoundError("未填写本地音频路径。")
        if not os.path.exists(p):
            raise FileNotFoundError(f"本地音频路径不存在：{p}")
        return p
    else:
        raise ValueError(f"未知的 source_mode：{source_mode}")


def redact(text: str) -> str:
    """极简脱敏示例：可按需替换为更复杂规则。"""
    return text.replace("sk-", "sk-*****")


def _limit_len(turns: List[str], max_chars: int) -> List[str]:
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


# ---------------------------
# LLM 抽取（OpenAI SDK）
# ---------------------------
def extract_profile(
    turns: List[str],
    schema: Dict[str, Any],
    system_prompt: str,
    model: str,
    base_url: str = None,
) -> Dict[str, Any]:
    """
    调用 OpenAI 兼容接口，强制 JSON 返回。
    若新 SDK (openai>=1.0.0)，优先使用 client.chat.completions；
    如网关仅支持老式端点，也能兼容。
    """
    try:
        from openai import OpenAI  # 官方 SDK v1+
    except Exception as e:
        raise RuntimeError("未安装 openai SDK，请先 pip install openai>=1.0.0") from e

    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)

    user_content = "请从以下转写文本中抽取客户画像信息，严格输出 JSON：\n\n" + "\n".join(turns)
    # 尝试 Chat Completions + JSON 模式
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},  # 需要支持 JSON mode 的模型
            temperature=0.1,
        )
        text = resp.choices[0].message.content
        data = json.loads(text)
        # 可选：简单 schema 字段校验（只校验顶层 key 存在）
        if isinstance(schema, dict) and "properties" in schema:
            for key in schema["properties"].keys():
                if key not in data:
                    # 不强制报错，只在缺失时补 None
                    data.setdefault(key, None)
        return data
    except Exception as e_json_mode:
        # 退回普通文本，再尝试解析 JSON（为兼容不支持 JSON mode 的网关）
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt + "\n务必仅输出 JSON 对象，不要包含其他文本。"},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()
        # 抽取最外层 JSON（简单做法：从第一个 { 到最后一个 }）
        l, r = text.find("{"), text.rfind("}")
        if l >= 0 and r >= 0 and r > l:
            text = text[l : r + 1]
        try:
            data = json.loads(text)
        except Exception as e_parse:
            raise TypeError(f"LLM 返回无法解析为 JSON。原文片段：{text[:200]} ...") from e_parse
        return data


# ---------------------------
# 主流程（供 UI/CLI 调用）
# ---------------------------
def analyze_once(
    audio,
    source_mode: str,
    manual_path: str,
    openai_key: str,
    openai_base_url: str,
    override_model: str,
):
    """
    返回：(asr_text, profile_json, status_text, debug_tail)
    - asr_text: ASR 文本
    - profile_json: 画像 JSON 字符串
    - status_text: 状态/提示/日志路径
    - debug_tail: 最近 N 行日志
    """
    logger, trace_id, log_path = _init_logger()
    _ui_log(logger, "INFO", f"Trace ID: {trace_id} | config_from={CONFIG_IMPORTED_FROM}")

    try:
        with _stage(logger, "目录准备"):
            _ensure_dirs()

        with _stage(logger, "输入音频准备"):
            audio_path = _prepare_audio_file(source_mode, getattr(audio, "name", None), manual_path)
            _ui_log(logger, "INFO", f"定位到音频：{audio_path}")

        with _stage(logger, "OpenAI Key 设置"):
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("未检测到 OPENAI_API_KEY。请在 UI 中填入或在系统环境中设置。")
            base_url = (openai_base_url or "").strip() or None
            if base_url and not (base_url.startswith("http://") or base_url.startswith("https://")):
                raise ValueError(f"Base URL 非法：{base_url}")

        with _stage(logger, "ASR 转写"):
            asr_model = getattr(config, "ASR_MODEL", "large-v3")
            asr_lang = getattr(config, "ASR_LANG", "zh")
            _ui_log(logger, "INFO", f"ASR 参数：model={asr_model}, lang={asr_lang}")
            turns = _asr_transcribe(audio_path, asr_model, asr_lang)
            if not turns:
                raise RuntimeError("ASR 未返回任何文本。请检查音频是否是可识别的语音片段。")
            _ui_log(logger, "INFO", f"ASR 行数：{len(turns)}")

        with _stage(logger, "脱敏与限长"):
            turns = [redact(t) for t in turns]
            max_chars = getattr(config, "MAX_CHARS_TO_LLM", 3000)
            before = sum(len(t) for t in turns)
            turns = _limit_len(turns, max_chars)
            after = sum(len(t) for t in turns)
            _ui_log(logger, "INFO", f"长度裁剪：{before} → {after} 字符")

        with _stage(logger, "Schema/Prompt 加载"):
            schema_path = os.path.join("schemas", "customer_profile.schema.json")
            prompt_path = os.path.join("prompts", "extract_profile_system.txt")
            if not os.path.exists(schema_path):
                raise FileNotFoundError(f"缺少 Schema：{schema_path}")
            if not os.path.exists(prompt_path):
                raise FileNotFoundError(f"缺少 Prompt：{prompt_path}")
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            with open(prompt_path, "r", encoding="utf-8") as f:
                sys_prompt = f.read()

        with _stage(logger, "LLM 抽取"):
            model_name = (override_model or "").strip() or getattr(config, "OPENAI_MODEL", "gpt-5")
            _ui_log(logger, "INFO", f"LLM 参数：model={model_name}, base_url={base_url or '默认'}")
            profile = extract_profile(turns=turns, schema=schema, system_prompt=sys_prompt,
                                      model=model_name, base_url=base_url)
            if not isinstance(profile, dict):
                raise TypeError("LLM 返回非 JSON 对象。")

        with _stage(logger, "结果保存"):
            base = os.path.splitext(os.path.basename(audio_path))[0]
            out_dir = getattr(config, "OUTPUT_DIR", "outputs/reports")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{base}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            _ui_log(logger, "INFO", f"输出已保存：{out_path}")

        # 汇总 UI 输出
        asr_text = "\n".join(turns)
        profile_json = json.dumps(profile, ensure_ascii=False, indent=2)
        status = (
            f"✅ 完成（Trace: {trace_id}）。\n日志文件：{log_path}\n"
            f"提示：如需查看详细栈信息，请打开日志文件。"
        )
        debug_tail = "\n".join(_UI_RING)
        return asr_text, profile_json, status, debug_tail

    except Exception as e:
        err = (
            f"❌ 失败（Trace: {trace_id}）：{e}\n"
            f"日志文件：{log_path}\n"
            f"请展开调试日志或打开日志文件定位失败阶段与堆栈。"
        )
        return "", "", err, "\n".join(_UI_RING)


# ---------------------------
# Gradio UI
# ---------------------------
def build_ui():
    import gradio as gr

    with gr.Blocks(title="Sales Call Analyzer", theme="soft") as demo:
        gr.Markdown("## 销售通话分析 - Pipeline (带阶段化日志)")

        with gr.Row():
            with gr.Column():
                source_mode = gr.Radio(
                    ["upload", "manual"], value="upload", label="音频来源 (upload=上传 / manual=本地路径)"
                )
                audio = gr.Audio(label="上传音频（选择 upload 模式）", type="filepath")
                manual_path = gr.Textbox(label="本地音频路径（选择 manual 模式）", placeholder="例如：data/raw/demo.wav")

                openai_key = gr.Textbox(label="OpenAI API Key", type="password")
                openai_base_url = gr.Textbox(label="自定义 Base URL（可选）", placeholder="留空使用默认官方网关")
                override_model = gr.Textbox(label="覆盖模型名（可选）", placeholder="留空使用 config.OPENAI_MODEL")

                analyze_btn = gr.Button("开始分析", variant="primary")

            with gr.Column():
                asr_out = gr.Textbox(label="ASR 转写文本", lines=10)
                profile_out = gr.Textbox(label="LLM 画像 JSON", lines=10)
                status_out = gr.Textbox(label="状态 / 提示", lines=5)

        with gr.Accordion("调试日志（最近 N 行）", open=False):
            debug_log = gr.Textbox(label="Log Tail", lines=12)
            refresh_btn = gr.Button("刷新日志")

        def _dump_ui_log():
            return "\n".join(_UI_RING)

        analyze_btn.click(
            fn=analyze_once,
            inputs=[audio, source_mode, manual_path, openai_key, openai_base_url, override_model],
            outputs=[asr_out, profile_out, status_out, debug_log],
        )
        refresh_btn.click(fn=_dump_ui_log, inputs=None, outputs=debug_log)

    return demo


# ---------------------------
# CLI
# ---------------------------
def run_cli(args):
    dummy_audio = type("Dummy", (), {"name": args.cli})  # 伪装成 gradio 的文件对象
    asr_text, profile_json, status, debug_tail = analyze_once(
        audio=dummy_audio,
        source_mode="manual",
        manual_path=args.cli,
        openai_key=args.key or "",
        openai_base_url=args.base_url or "",
        override_model=args.model or "",
    )
    print(status)
    if asr_text:
        print("\n=== ASR ===")
        print(asr_text)
    if profile_json:
        print("\n=== PROFILE ===")
        print(profile_json)
    if debug_tail:
        print("\n=== LOG TAIL ===")
        print(debug_tail)


# ---------------------------
# 入口
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Call Analyzer")
    parser.add_argument("--cli", type=str, help="以 CLI 模式分析指定音频文件（跳过 GUI）")
    parser.add_argument("--key", type=str, help="OpenAI API Key（CLI 模式可选）")
    parser.add_argument("--base_url", type=str, help="自定义 OpenAI Base URL（CLI 模式可选）")
    parser.add_argument("--model", type=str, help="覆盖模型名（CLI 模式可选）")
    args = parser.parse_args()

    if args.cli:
        run_cli(args)
    else:
        ui = build_ui()
        ui.launch(server_name=getattr(config, "GRADIO_SERVER_NAME", "127.0.0.1"),
                  server_port=getattr(config, "GRADIO_SERVER_PORT", 7860))
