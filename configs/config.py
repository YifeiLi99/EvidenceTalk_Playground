"""
config.py — 完全替换版（保持轻量 & 相对路径）
注意：所有路径均为相对路径，便于跨平台移动。
"""

# 目录
AUDIO_DIR = "data/raw"
OUTPUT_DIR = "outputs/reports"
LOG_DIR = "outputs/logs"

# ASR
ASR_MODEL = "large-v3"   # 显存紧张可改为 "medium" / "small"
ASR_LANG = "zh"

# LLM
OPENAI_MODEL = "gpt-5"   # 可在 UI 覆盖；或 CLI --model 覆盖
OPENAI_BASE_URL = None   # 留空使用官方；也可在 UI 覆盖

# 控制
MAX_CHARS_TO_LLM = 3000
MAX_LOG_LINES_SHOWN = 200

# Gradio
GRADIO_SERVER_NAME = "127.0.0.1"
GRADIO_SERVER_PORT = 7860
