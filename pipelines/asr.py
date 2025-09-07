"""
asr.py — 完全替换版
特性：
- 优先 CUDA + float16；失败则自动降级到 CPU + int8
- 明确打印 device / compute_type / 耗时
- 语句切分：按中英文句末符号做粗切，保留 [T1]、[T2] 标号格式
"""

import os
import time
import re
from typing import List
from faster_whisper import WhisperModel


def transcribe(audio_path: str, model_name: str = "large-v3", lang: str = "zh") -> List[str]:
    t0 = time.time()

    device = "cuda"
    compute = "float16"
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute)
    except Exception:
        # 明确降级提示
        device = "cpu"
        compute = "int8"  # CPU 上更省内存
        model = WhisperModel(model_name, device=device, compute_type=compute)

    print(f"[ASR] device={device}, compute_type={compute}, model={model_name}, lang={lang}")

    # VAD 开启，提升转写稳定性；若报错可将 vad_filter=False
    segments, _info = model.transcribe(audio_path, language=lang, vad_filter=True)

    # 收集文本
    text = "".join(s.text for s in segments).strip()

    # 以中英文常见句末符号做粗切
    parts = [p.strip() for p in re.split(r"[。！？!?；;]\s*", text) if p.strip()]

    turns = [f"[T{i+1}] {p}" for i, p in enumerate(parts)]
    print(f"[ASR] segments={getattr(segments, '__len__', lambda: '?')()} | "
          f"sentences={len(turns)} | time={(time.time() - t0) * 1000:.1f} ms")
    return turns
