"""
配置
"""

# 目录
AUDIO_DIR = "data/raw"
OUTPUT_DIR = "outputs/reports"
LOG_DIR = "outputs/logs"

# ASR
ASR_MODEL = "large-v3"   # 显存紧张可改为 "medium" / "small"
ASR_LANG = "zh"

# LLM
OPENAI_MODEL = "gpt-5-mini"   # 自由调整
OPENAI_BASE_URL = None   # 留空使用官方；也可在 UI 覆盖

# 控制
MAX_CHARS_TO_LLM = 3000
MAX_LOG_LINES_SHOWN = 200

# Gradio
GRADIO_SERVER_NAME = "127.0.0.1"
GRADIO_SERVER_PORT = 7860
