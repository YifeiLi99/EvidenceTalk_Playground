"""
极简测试版
"""

import os
from typing import Optional
from openai import OpenAI
import gradio as gr
from gradio.themes import Soft

DEFAULT_MODEL = "gpt-5-mini"  # 如需改模型，改这里即可

SYSTEM_PROMPT = """你是一名销售助理。你会接收一段通过麦克风转录得到的客户对话文本。
请只进行一次性分析，不要反问或进行多轮互动。
请根据客户的表述，尽可能提取其画像信息，输出需简洁、结构化、要点清晰，默认使用中文。
"""

USER_ANALYSIS_INSTR = """请分析以下销售对话文本，推测并总结客户的关键信息，包括但不限于：
1) 客户类型（如学生、上班族、个体户、企业主等）
2) 生活方式（如消费习惯、兴趣偏好、日常作息等）
3) 家庭状况（如是否已婚、有无子女、家庭成员特点等）
4) 收入情况（如收入水平、大致来源、消费能力等）
如果某项信息无法从文本中确定，请标注为“未知”。
文本：
"""

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
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ 调用失败：{type(e).__name__}: {e}"

def build_ui():
    with gr.Blocks(title="最简文本分析 Demo", theme=Soft()) as demo:
        gr.Markdown("## 最简文本分析 Demo\n只输入文本与 API Key，一次调用 GPT 输出结果。")

        with gr.Row():
            with gr.Column():
                api_key = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...")
                text_in = gr.Textbox(label="输入文本", lines=8, placeholder="把要分析的文本粘贴到这里")
                btn = gr.Button("开始分析", variant="primary")
            with gr.Column():
                out = gr.Textbox(label="分析结果", lines=14)

        def on_click(t, k):
            return run_once(t, k)

        btn.click(fn=on_click, inputs=[text_in, api_key], outputs=out)

    return demo

if __name__ == "__main__":
    ui = build_ui()
    try:
        # 默认尝试本地访问
        ui.launch()
    except ValueError as e:
        # 若环境不允许 localhost，自动回退为可分享链接
        msg = str(e)
        if "localhost is not accessible" in msg or "shareable link must be created" in msg:
            print("⚠️ 本地回环不可达，自动使用 share=True 重启。")
            ui.launch(share=True)
        else:
            raise
