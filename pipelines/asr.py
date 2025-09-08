"""
asr模块（稳字优先强化版）
"""

# 【改动点-0】新增依赖/模块
import os
import time
import re
from typing import List, Tuple, Optional, Dict
from faster_whisper import WhisperModel

# ============ 配置区域（可按需调整）============

# 【改动点-1】相对路径：词表文件（每行一个词/短语）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEXICON_PATH = os.path.join(BASE_DIR, "prompts", "lexicon.txt")

# 【改动点-2】常错词 → 正确词 的强制替换字典（遇到就直接替换，后面还有更轻量的模糊纠错兜底）
#   这里放你反馈的典型误识别。你后续可随时扩充。
HARD_REPLACE: Dict[str, str] = {
    "子野姐": "理解",
    "分胎": "分开",
    "外骑":"外勤",
    "值吧":"值班",
    "刺卡":"次卡",
    "靠电":"靠垫"
    # 你可以继续加： "某错词": "正确词",
}

# 【改动点-3】ASR 过滤阈值（更稳）
AVG_LOGPROB_DROP_TH = -1.2      # 段平均对数概率过低则丢弃
NO_SPEECH_DROP_TH = 0.6         # 静音概率过高则丢弃
LOW_WORD_PROB_TH = 0.50         # 词级置信度低于此阈值，用 «...» 标记

# 【改动点-4】解码参数（慢但稳）
DECODE_OPTS = dict(
    beam_size=8,                # ↑搜索更充分
    best_of=8,                  # ↑N-best内部采样
    patience=1.2,               # 更耐心
    temperature=[0.0, 0.2, 0.4],# 低温度先搜，减少幻听
)

VAD_PARAMS = dict(
    min_silence_duration_ms=400 # 稍强的静音切分
)

# ============ 工具函数 ============

# 【改动点-5】加载词表，拼成 initial_prompt（限制长度，避免过长）
def _load_lexicon(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def _build_initial_prompt(lexicon: List[str], max_chars: int = 400) -> Optional[str]:
    if not lexicon:
        return None
    joined = " ".join(lexicon)
    return joined[:max_chars]

# 【改动点-6】强制替换 + 可选模糊纠错（rapidfuzz 可选）
def _post_correct_text(text: str, lexicon: List[str]) -> str:
    # 1) 先做强制映射（不会触碰 «...» 中的不可靠词）
    def hard_replace_safe(s: str) -> str:
        # 保护 «...»：split-join 保留不可靠片段原样
        out = []
        parts = s.split("«")
        for idx, part in enumerate(parts):
            if idx == 0:
                # outside unreliable
                out.append(_hard_replace_plain(part))
            else:
                unreliable, rest = part.split("»", 1) if "»" in part else (part, "")
                out.append("«" + unreliable + "»" + _hard_replace_plain(rest))
        return "".join(out)

    def _hard_replace_plain(s: str) -> str:
        for wrong, right in HARD_REPLACE.items():
            s = s.replace(wrong, right)
        return s

    text = hard_replace_safe(text)

    # 2) 可选：基于 lexicon 的轻量纠错（若 rapidfuzz 不存在则跳过）
    try:
        from rapidfuzz import process, fuzz
    except Exception:
        return text

    # 简单策略：仅当整句与任何词条的 partial_ratio 极高时替换（非常保守）
    # 生产环境可换成“滑窗+词表”或“分词+近邻替换”的策略，这里保持轻量低风险
    for term in lexicon:
        if 2 <= len(term) <= 6:
            match = process.extractOne(term, [text], scorer=fuzz.partial_ratio)
            if match and match[1] >= 95:
                text = text.replace(match[0], term)  # 近似段替换为词表项
    return text

# ============ 模型加载（本地优先 + 惰性）===========

# 【改动点-7】支持本地目录优先；环境变量 ASR_LOCAL_MODEL_DIR 覆盖
_MODEL = None
_MODEL_PATH = None

def _load_model(model_name_or_dir: str, device: str, compute: str) -> WhisperModel:
    return WhisperModel(model_name_or_dir, device=device, compute_type=compute)

def _get_model(model_name: str) -> Tuple[WhisperModel, str, str, str]:
    """
    返回：(model, device, compute, model_path_used)
    - 若传入目录存在，则直接加载该目录；
    - 否则使用传入的模型名字（会走 faster-whisper 的本地缓存；若首次运行可能尝试联网，建议提前准备本地CT2目录并把 model_name 指向它）。
    - 若 GPU 失败则自动降级到 CPU/INT8。
    """
    global _MODEL, _MODEL_PATH
    if _MODEL is not None:
        return _MODEL, _MODEL_DEVICE, _MODEL_COMPUTE, _MODEL_PATH

    prefer = None
    if model_name and os.path.isdir(model_name):
        prefer = model_name
    else:
        env_dir = os.getenv("ASR_LOCAL_MODEL_DIR")
        if env_dir and os.path.isdir(env_dir):
            prefer = env_dir

    device, compute = "cuda", "float16"
    try:
        model_path = prefer if prefer else model_name
        model = _load_model(model_path, device, compute)
    except Exception:
        device, compute = "cpu", "int8"
        model_path = prefer if prefer else model_name
        model = _load_model(model_path, device, compute)

    # 缓存
    _cache_model(model, device, compute, model_path)
    return _MODEL, _MODEL_DEVICE, _MODEL_COMPUTE, _MODEL_PATH

def _cache_model(m, device, compute, path):
    global _MODEL, _MODEL_DEVICE, _MODEL_COMPUTE, _MODEL_PATH
    _MODEL = m
    _MODEL_DEVICE = device
    _MODEL_COMPUTE = compute
    _MODEL_PATH = path

# ============ 主流程（稳字优先）===========

# 【改动点-8】新增：返回“整段文本 + 质量报告”的接口（推荐在 UI 公屏展示）
def transcribe_with_conf(
    audio_path: str,
    model_name: str = "large-v3",
    lang: str = "zh",
    low_word_prob: float = LOW_WORD_PROB_TH,
) -> Tuple[str, dict]:
    t0 = time.time()
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在：{audio_path}")

    # 模型加载（本地优先 + 自动降级）
    model, device, compute, model_path = _get_model(model_name)

    # 词表→initial_prompt
    lexicon = _load_lexicon(LEXICON_PATH)
    init_prompt = _build_initial_prompt(lexicon)

    # 转写（更稳的参数）
    segments, info = model.transcribe(
        audio_path,
        language=lang,
        vad_filter=True,
        vad_parameters=VAD_PARAMS,
        word_timestamps=True,              # 词级置信度
        condition_on_previous_text=True,   # 按上下文延续，减少突变
        initial_prompt=init_prompt,        # 关键词引导
        **DECODE_OPTS
    )

    kept_lines, dropped = [], 0
    avg_logs = []
    for seg in segments:
        avg_lp = getattr(seg, "avg_logprob", 0.0)
        if avg_lp < AVG_LOGPROB_DROP_TH or getattr(seg, "no_speech_prob", 0.0) > NO_SPEECH_DROP_TH:
            dropped += 1
            continue
        avg_logs.append(avg_lp)

        # 词级置信度标注
        if seg.words:
            buf = []
            for w in seg.words:
                token = (w.word or "").strip()
                if not token:
                    continue
                prob = getattr(w, "probability", 1.0)
                if prob is not None and prob < low_word_prob:
                    buf.append(f"«{token}»")
                else:
                    buf.append(token)
            line = "".join(buf).strip()
        else:
            line = seg.text.strip()

        if line:
            kept_lines.append(line)

    # 合并文本 & 轻量纠错
    raw_text = "。".join(kept_lines).replace("。。", "。").strip("。")
    corrected = _post_correct_text(raw_text, lexicon)

    report = {
        "device": device,
        "compute_type": compute,
        "model_path": model_path,
        "avg_logprob_mean": round(sum(avg_logs)/len(avg_logs), 3) if avg_logs else None,
        "segments_dropped": dropped,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "low_word_prob_threshold": low_word_prob,
        "lexicon_used": bool(lexicon),
        "lexicon_size": len(lexicon),
    }
    return corrected, report

# 【改动点-9】兼容旧接口：仍返回逐句列表 + 带 [T#] 编号
def transcribe(audio_path: str, model_name: str = "large-v3", lang: str = "zh") -> List[str]:
    text, report = transcribe_with_conf(audio_path, model_name=model_name, lang=lang)
    parts = [p.strip() for p in re.split(r"[。！？!?；;]\s*", text) if p.strip()]
    return [f"[T{i+1}] {p}" for i, p in enumerate(parts)]
