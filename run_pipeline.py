"""
正式版（不进行json化）
"""
import os
from typing import Optional, Tuple
from openai import OpenAI
import gradio as gr
from configs.config import OPENAI_MODEL
from pipelines.asr import transcribe

def load_prompts(path: str):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = content.split("[USER_ANALYSIS_INSTR]")
    system_prompt = parts[0].replace("[SYSTEM_PROMPT]", "").strip()
    user_instr = parts[1].strip()
    return system_prompt, user_instr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(BASE_DIR, "prompts", "extract_profile_system.txt")
SYSTEM_PROMPT, USER_ANALYSIS_INSTR = load_prompts(PROMPT_FILE)

def run_once(text: str, api_key: Optional[str]) -> str:
    if not text or not text.strip():
        return "❌ 请输入要分析的文本。"
    if not api_key or not api_key.strip():
        return "❌ 请输入有效的 OpenAI API Key。"
    client = OpenAI(api_key=api_key.strip(), timeout=60.0)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_ANALYSIS_INSTR + text.strip()},
    ]
    try:
        resp = client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ 调用失败：{type(e).__name__}: {e}"

# 稳健解析 gr.File 的值为“本地文件路径”
def _resolve_file_to_path(f) -> Optional[str]:
    # gr.File 可能传 dict、临时文件对象或直接字符串
    if f is None:
        return None
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        # gradio v4 常见结构：{"name": ".../tmp/xxx.wav", "size":..., "orig_name":"xxx.wav", ...}
        return f.get("name") or f.get("path") or f.get("orig_name")
    # 临时文件对象
    name = getattr(f, "name", None)
    if isinstance(name, str):
        return name
    return None

def asr_then_analyze(file_obj, api_key: Optional[str],
                     asr_model: str = "large-v3", asr_lang: str = "zh") -> Tuple[str, str]:
    audio_path = _resolve_file_to_path(file_obj)
    if not audio_path:
        return "❌ 请先上传 WAV 音频。", ""
    if not os.path.exists(audio_path):
        return f"❌ 找不到音频文件：{audio_path}", ""
    if not api_key or not api_key.strip():
        return "❌ 请输入有效的 OpenAI API Key。", ""
    try:
        turns = transcribe(audio_path, model_name=asr_model, lang=asr_lang)  # 你的 asr.py
        asr_text = "\n".join(turns).strip() if turns else ""
        if not asr_text:
            return "⚠️ ASR 未识别到有效文本。", ""
    except Exception as e:
        return f"❌ ASR 失败：{type(e).__name__}: {e}", ""
    llm_result = run_once(asr_text, api_key)
    return asr_text, llm_result

def build_ui():
    #去掉 theme=Soft()，只用最基础 Blocks
    with gr.Blocks(title="ASR → 文本画像分析 Demo") as demo:
        gr.Markdown(
            "## ASR → 文本画像分析 Demo\n"
            "- 上传 WAV → 点击按钮 → 自动转写并调用 LLM 输出结构化分析。\n"
            "- 提示词从 `prompts/extract_profile_system.txt` 读取。"
        )
        with gr.Row():
            with gr.Column():
                api_key = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...")
                # gr.File 仅保留最小参数，避免 schema 分支触发
                audio_in = gr.Textbox(
                    label="音频绝对路径（只要 .wav）",
                    placeholder=r"C:\path\to\your.wav"
                )
                with gr.Accordion("ASR 参数（可选）", open=False):
                    asr_model = gr.Textbox(value="large-v3", label="Whisper 模型名")
                    asr_lang = gr.Textbox(value="zh", label="语言代码")
                btn = gr.Button("转写并分析", variant="primary")
            with gr.Column():
                asr_out = gr.Textbox(label="ASR 转写结果", lines=10)
                llm_out = gr.Textbox(label="LLM 分析结果", lines=14)

        def on_click(path_str, k, m, lang):
            return asr_then_analyze_path(path_str, k, asr_model=m, asr_lang=lang)
        btn.click(fn=on_click, inputs=[audio_in, api_key, asr_model, asr_lang], outputs=[asr_out, llm_out])

        def asr_then_analyze_path(audio_path: Optional[str], api_key: Optional[str],
                                  asr_model: str = "large-v3", asr_lang: str = "zh"):
            if not audio_path or not audio_path.strip():
                return "❌ 请先填入 WAV 的绝对路径。", ""
            audio_path = audio_path.strip().strip('"')
            if not os.path.exists(audio_path):
                return f"❌ 找不到音频文件：{audio_path}", ""
            if not audio_path.lower().endswith(".wav"):
                return f"❌ 仅支持 .wav 文件：{audio_path}", ""
            if not api_key or not api_key.strip():
                return "❌ 请输入有效的 OpenAI API Key。", ""

            try:
                turns = transcribe(audio_path, model_name=asr_model, lang=asr_lang)
                asr_text = "\n".join(turns).strip() if turns else ""
                if not asr_text:
                    return "⚠️ ASR 未识别到有效文本。", ""
            except Exception as e:
                return f"❌ ASR 失败：{type(e).__name__}: {e}", ""

            llm_result = run_once(asr_text, api_key)
            return asr_text, llm_result

    return demo

if __name__ == "__main__":
    ui = build_ui()
    try:
        ui.launch()
    except ValueError as e:
        msg = str(e)
        if "localhost is not accessible" in msg or "shareable link must be created" in msg:
            print("⚠️ 本地回环不可达，自动使用 share=True 重启。")
            ui.launch(share=True)
        else:
            raise
