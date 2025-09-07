AUDIO_DIR = "data/raw"
OUTPUT_DIR = "outputs/reports"
LOG_DIR = "outputs/logs"

ASR_MODEL = "large-v3"      # 卡不够就用 "medium"
ASR_LANG = "zh"

OPENAI_MODEL = "gpt-5"      # 你要用的模型名
OPENAI_BASE_URL = None      # 如需自定义网关就填，否则留 None

MAX_CHARS_TO_LLM = 3000     # 防超长，首次就粗暴限长
