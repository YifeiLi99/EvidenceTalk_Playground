# 只负责：把 data/raw/xxx.wav 变成一串简洁句子列表
from faster_whisper import WhisperModel
import re

def transcribe(audio_path, model_name, lang):
    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    segments, _ = model.transcribe(audio_path, language=lang, vad_filter=True)
    text = "".join(s.text for s in segments).strip()
    # 简单切句：按 。！？ 分段，并加行号 [T1]…
    sents = [s for s in re.split(r'[。！？!?]\s*', text) if s]
    turns = [f"[T{i+1}] {s}" for i, s in enumerate(sents)]
    return turns
