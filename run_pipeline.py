# run_pipeline.py
# =============================================================================
# [NEW] 2025-09-07 彻底改造成带 Gradio 的交互式入口：
#  - 打开即弹 Gradio 窗口（录音 / 上传 / 本地路径 三选一）
#  - 手动设置 OpenAI API Key（和可选 Base URL）
#  - 点击“分析”后，下方同时展示：ASR 转写文本 + LLM 画像 JSON
#  - 仍然保留一个极简 CLI 兜底模式：python run_pipeline.py --cli data/raw/demo.wav
#  - 全部使用相对路径，路径与开关集中在 configs/config.py
# =============================================================================

# -------------------------[NEW | 保留与扩展的导入]----------------------------
import os
import sys  # [NEW] 为了支持 --cli 兜底模式
import json
import time   # [NEW] 用于生成会话文件名
import shutil # [NEW] 复制输入音频到 data/raw 统一管理
import traceback  # [NEW] 友好错误展示
import argparse   # [NEW] CLI 模式参数解析

import gradio as gr  # [NEW] Gradio GUI
from configs import config
from pipelines.asr import transcribe
from pipelines.redact import redact
from pipelines.extract_ie import extract_profile

# -------------------------[NEW | 小工具函数]-----------------------------------
def _ensure_dirs():
    """[NEW] 确保输出与数据目录存在（相对路径）"""
    os.makedirs(getattr(config, "AUDIO_DIR", "data/raw"), exist_ok=True)
    os.makedirs(getattr(config, "OUTPUT_DIR", "outputs/reports"), exist_ok=True)
    os.makedirs(getattr(config, "LOG_DIR", "outputs/logs"), exist_ok=True)

def _prepare_audio_file(source_mode: str, mic_or_upload_path: str, manual_path: str) -> str:
    """
    [NEW] 统一产出一个可供 pipeline 使用的本地音频路径：
      - source_mode: "录音/上传" 或 "本地路径"
      - mic_or_upload_path: gr.Audio 返回的路径（type="filepath"）
      - manual_path: 文本框输入的路径（绝对/相对均可）
    逻辑：复制源文件到 data/raw 下，命名为 session_时间戳.wav
    """
    audio_dir = getattr(config, "AUDIO_DIR", "data/raw")
    ts = time.strftime("%Y%m%d-%H%M%S")
    target = os.path.join(audio_dir, f"session_{ts}.wav")

    if source_mode == "录音/上传":
        if not mic_or_upload_path or not os.path.exists(mic_or_upload_path):
            raise FileNotFoundError("请先录音或上传音频文件。")
        shutil.copyfile(mic_or_upload_path, target)
        return target

    # 本地路径模式
    path = (manual_path or "").strip()
    if not path:
        raise FileNotFoundError("请输入本地音频文件路径。")
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到音频文件：{path}")
    shutil.copyfile(path, target)
    return target

def _limit_len(turns, max_chars: int):
    """[NEW] 简单长度裁剪，避免一次性请求过长"""
    total = 0
    kept = []
    for t in turns:
        total += len(t)
        kept.append(t)
        if total >= max_chars:
            break
    return kept

# -------------------------[NEW | 核心分析函数]---------------------------------
def analyze_once(
    audio_file_path: str,
    source_mode: str,
    manual_path: str,
    openai_key: str,
    openai_base_url: str,
    override_model: str = ""
):
    """
    [NEW] 执行一次完整分析：
      1) 准备音频文件（复制到 data/raw）
      2) 设置 OPENAI_API_KEY（如果用户输入）
      3) ASR 转写（segments→句子→行号）
      4) 脱敏 + 限长
      5) 载入 schema 与提示词
      6) LLM 抽取（JSON）
      7) 保存 JSON 并回显
    返回：(asr_text, profile_json, status_message)
    """
    try:
        _ensure_dirs()

        # [NEW] 1) 准备音频
        audio_path = _prepare_audio_file(source_mode, audio_file_path, manual_path)

        # [NEW] 2) 设置 OpenAI Key
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        elif not os.getenv("OPENAI_API_KEY"):
            return "", "", "❌ 未检测到 OpenAI Key：请在页面填写或预先设置环境变量。"

        base_url = (openai_base_url or "").strip() or None

        # [NEW] 3) ASR 转写
        asr_model = getattr(config, "ASR_MODEL", "large-v3")
        asr_lang = getattr(config, "ASR_LANG", "zh")
        turns = transcribe(audio_path, asr_model, asr_lang)

        # [NEW] 4) 脱敏 + 限长
        turns = [redact(t) for t in turns]
        max_chars = getattr(config, "MAX_CHARS_TO_LLM", 3000)
        turns = _limit_len(turns, max_chars)

        # [NEW] 5) 载入 schema 与系统提示
        with open("schemas/customer_profile.schema.json", "r", encoding="utf-8") as f:
            schema = json.load(f)
        with open("prompts/extract_profile_system.txt", "r", encoding="utf-8") as f:
            sys_prompt = f.read()

        # [NEW] 6) LLM 抽取
        model_name = (override_model.strip() or getattr(config, "OPENAI_MODEL", "gpt-5"))
        profile = extract_profile(
            turns=turns,
            schema=schema,
            system_prompt=sys_prompt,
            model=model_name,
            base_url=base_url
        )

        # [NEW] 7) 保存结果
        base = os.path.splitext(os.path.basename(audio_path))[0]
        out_dir = getattr(config, "OUTPUT_DIR", "outputs/reports")
        out_path = os.path.join(out_dir, f"{base}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        # 展示
        asr_text = "\n".join(turns)
        profile_json = json.dumps(profile, ensure_ascii=False, indent=2)
        status = f"✅ 分析完成：{os.path.basename(audio_path)}  →  保存于 {out_path}"
        return asr_text, profile_json, status

    except Exception as e:
        err = f"❌ 发生错误：{e}\n{traceback.format_exc()}"
        return "", "", err

# -------------------------[NEW | Gradio UI 构建]-------------------------------
def build_ui():
    with gr.Blocks(title="EvidenceTalk-Lab | 实验版") as demo:
        gr.Markdown("## EvidenceTalk-Lab（实验版）\n"
                    "选择输入方式 → 填 OpenAI Key → 点击 **开始分析** → 下方显示 ASR 与画像 JSON。")

        with gr.Row():
            source_mode = gr.Radio(
                choices=["录音/上传", "本地路径"],
                value="录音/上传",
                label="输入来源"
            )

        with gr.Row(variant="panel"):
            audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="录音或上传音频（支持 wav/flac/mp3）",
            )
            manual_path = gr.Textbox(
                label="或：本地音频文件路径（绝对/相对）",
                placeholder="例如：data/raw/demo.wav 或 C:\\path\\to\\demo.wav"
            )

        with gr.Row():
            openai_key = gr.Textbox(
                type="password",
                label="OpenAI API Key（仅本地使用，不会外传）",
                placeholder="sk-xxxxx"
            )
            openai_base_url = gr.Textbox(
                label="可选：自定义 OpenAI Base URL",
                placeholder="留空则使用官方默认"
            )

        with gr.Row():
            override_model = gr.Textbox(
                label="可选：临时覆盖模型名（不填则用 configs/config.py 的 OPENAI_MODEL）",
                placeholder="例如：gpt-4o-mini / qwen2.5-7b-instruct（经网关）"
            )
            analyze_btn = gr.Button("开始分析", variant="primary")

        with gr.Row():
            asr_out = gr.Textbox(label="ASR 转写（脱敏，带行号）", lines=12)
            profile_out = gr.Code(label="画像 JSON（结构化抽取结果）", language="json")

        status_out = gr.Markdown()

        # 交互：开始分析
        analyze_btn.click(
            fn=analyze_once,
            inputs=[audio, source_mode, manual_path, openai_key, openai_base_url, override_model],
            outputs=[asr_out, profile_out, status_out]
        )

        # 根据输入来源切换可见性（提示友好）
        def toggle_inputs(mode):
            return (
                gr.update(visible=True),                   # audio 组件总是可见（录音/上传）
                gr.update(visible=(mode == "本地路径")),   # 仅本地路径模式下显示路径输入框
            )
        source_mode.change(
            toggle_inputs,
            inputs=[source_mode],
            outputs=[audio, manual_path]
        )

    return demo

# -------------------------[NEW | CLI 兜底模式]--------------------------------
def run_cli_once(audio_path: str, api_key: str = "", base_url: str = "", model: str = ""):
    """[NEW] 简易 CLI：python run_pipeline.py --cli data/raw/demo.wav --key sk-xxx"""
    asr, js, status = analyze_once(
        audio_file_path=audio_path,
        source_mode="本地路径",
        manual_path=audio_path,
        openai_key=api_key,
        openai_base_url=base_url,
        override_model=model
    )
    print(status)
    if asr:
        print("\n=== ASR ===\n", asr)
    if js:
        print("\n=== PROFILE ===\n", js)

# -------------------------[NEW | 入口逻辑]-------------------------------------
if __name__ == "__main__":
    # 支持两种入口：
    #  - 默认：Gradio GUI
    #  - 兜底：CLI（--cli <audio_path>）
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", help="以 CLI 模式运行：指定音频路径", default=None)
    parser.add_argument("--key", help="OpenAI API Key（可选，CLI 用）", default="")
    parser.add_argument("--base", help="OpenAI Base URL（可选，CLI 用）", default="")
    parser.add_argument("--model", help="覆盖模型名（可选，CLI 用）", default="")
    args = parser.parse_args()

    if args.cli:
        _ensure_dirs()
        run_cli_once(args.cli, args.key, args.base, args.model)
    else:
        _ensure_dirs()
        app = build_ui()
        # [NEW] 直接启动界面
        app.launch()
