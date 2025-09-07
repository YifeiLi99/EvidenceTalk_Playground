"""
环境自检
"""

import sys
import os
import platform
import argparse
import subprocess
import importlib
from typing import Optional

# ---------- 工具函数 ----------
def print_kv(key: str, val: object):
    print(f"{key:<28} {val}")

def check_cmd_exists(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "--version"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True)
        return True
    except Exception:
        return False

def run_cmd_lines(cmd: list[str]) -> list[str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return out.decode(errors="ignore").splitlines()
    except Exception:
        return []

def import_optional(name: str):
    try:
        mod = importlib.import_module(name)
        return mod, None
    except Exception as e:
        return None, e

def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

# ---------- 检测模块 ----------
def diagnose_system():
    section("系统与 Python 基本信息")
    print_kv("Python", sys.version.split()[0])
    print_kv("Python 可执行文件", sys.executable)
    print_kv("平台", f"{platform.system()} {platform.release()} ({platform.machine()})")

    # nvidia-smi（若存在）
    if check_cmd_exists("nvidia-smi"):
        lines = run_cmd_lines(["nvidia-smi"])
        if lines:
            print_kv("NVIDIA-SMI 检测", "可用")
            # 打印前 3 行概览（驱动版本/显卡）
            for ln in lines[:3]:
                print("  " + ln.strip())
        else:
            print_kv("NVIDIA-SMI 检测", "命令存在但输出异常")
    else:
        print_kv("NVIDIA-SMI 检测", "不可用（可能是笔记本核显或驱动未装）")

def diagnose_torch():
    section("PyTorch / CUDA / cuDNN")
    torch, err = import_optional("torch")
    if not torch:
        print("未找到 PyTorch。示例安装（cu128）：")
        print("  pip install 'torch==2.7.0' 'torchaudio==2.7.0' --index-url https://download.pytorch.org/whl/cu128")
        return

    print_kv("torch 版本", torch.__version__)
    print_kv("CUDA runtime (torch.version.cuda)", getattr(torch.version, "cuda", None))
    cuda_ok = torch.cuda.is_available()
    print_kv("CUDA 可用 (torch)", cuda_ok)
    try:
        cudnn_v = torch.backends.cudnn.version()
    except Exception:
        cudnn_v = None
    print_kv("cuDNN 版本 (torch)", cudnn_v)
    if cuda_ok:
        try:
            print_kv("GPU 数量", torch.cuda.device_count())
            print_kv("当前 GPU", torch.cuda.current_device())
            print_kv("GPU 名称", torch.cuda.get_device_name(0))
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print_kv("GPU 显存 (GB)", f"{total_mem:.2f}")
        except Exception as e:
            print_kv("GPU 信息获取异常", repr(e))
    else:
        print("⚠️  Torch 未检测到 CUDA。请检查 NVIDIA 驱动与对应 CUDA 运行时是否就绪。")

def diagnose_ctranslate2_and_fw():
    section("CTranslate2 / faster-whisper")
    ct2, e1 = import_optional("ctranslate2")
    fw, e2 = import_optional("faster_whisper")

    if ct2:
        print_kv("ctranslate2 版本", getattr(ct2, "__version__", "unknown"))
        # 4.6.0 的稳定 API：GPU 数量 & 支持的计算类型
        cuda_cnt = 0
        try:
            cuda_cnt = ct2.get_cuda_device_count()
        except Exception as e:
            print_kv("get_cuda_device_count 异常", repr(e))
        print_kv("CT2 可见 CUDA 设备数", cuda_cnt)

        try:
            sup_types = ct2.get_supported_compute_types()
        except Exception as e:
            sup_types = []
            print_kv("get_supported_compute_types 异常", repr(e))
        print_kv("CT2 支持计算类型", sup_types)
        if cuda_cnt == 0:
            print("⚠️  CTranslate2 未检测到 GPU。常见原因：")
            print("   - 安装的是 CPU 轮子；请改装 CUDA 轮子（与系统 CUDA 对齐，例如 cu128）。")
            print("   - NVIDIA 驱动 / CUDA runtime / cuDNN 版本不匹配。")
    else:
        print("未安装 ctranslate2：", e1)

    if fw:
        print_kv("faster-whisper 版本", getattr(fw, "__version__", "unknown"))
    else:
        print("未安装 faster-whisper：", e2)

def diagnose_openai_and_whisper():
    section("OpenAI SDK / 原版 whisper / ffmpeg")
    openai, e_openai = import_optional("openai")
    if openai:
        print_kv("openai 版本", getattr(openai, "__version__", "unknown"))
    else:
        print("未安装 openai：", e_openai)

    whisper, e_whisper = import_optional("whisper")
    if whisper:
        print_kv("openai-whisper 版本", getattr(whisper, "__version__", "unknown"))
        has_ffmpeg = check_cmd_exists("ffmpeg")
        print_kv("ffmpeg 可用", has_ffmpeg)
        if not has_ffmpeg:
            print("⚠️ 未检测到 ffmpeg。原版 whisper 的音频处理需要它。")
    else:
        print("未安装原版 whisper（openai-whisper 包）：", e_whisper)

def run_minimal_inference(model_name: str, device_pref: str, compute_type: str, audio: Optional[str]):
    section("最小推理验证（faster-whisper）")
    fw, e_fw = import_optional("faster_whisper")
    if not fw:
        print("无法进行推理：未安装 faster-whisper：", e_fw)
        return

    # 创建模型并读取内部 CT2 状态
    try:
        model = fw.WhisperModel(
            model_name,
            device=device_pref,          # auto / cuda / cpu
            compute_type=compute_type    # float32 / float16 / int8_float16 / int8 / ...
        )
        # 读取内部 ctranslate2.models.Whisper（若版本不同可能字段名变化）
        inner = getattr(model, "_model", None)
        if inner is not None:
            dev = getattr(inner, "device", "unknown")
            dev_idx = getattr(inner, "device_index", "unknown")
            ctype = getattr(inner, "compute_type", "unknown")
            print_kv("Inner CT2 device", dev)
            print_kv("Inner CT2 device_index", dev_idx)
            print_kv("Inner CT2 compute_type", ctype)
        else:
            # 退化读取（有些版本可能没有 _model）
            dev = getattr(model, "device", None) or getattr(model, "_device", "unknown")
            ctype = getattr(model, "compute_type", None) or getattr(model, "_compute_type", "unknown")
            print_kv("WhisperModel 设备(回退)", dev)
            print_kv("compute_type(回退)", ctype)

        if str(dev).lower().find("cuda") == -1 and str(dev).lower().find("gpu") == -1:
            print("⚠️ 看起来没有在 GPU 上运行。若预期用 GPU：")
            print("   1) 确保安装 CUDA 版 ctranslate2（与系统 CUDA 对齐，如 cu128）")
            print("   2) 运行时使用 --device cuda（或 device='cuda'）")
            print("   3) 检查 NVIDIA 驱动 / CUDA / cuDNN 与 CT2 轮子是否匹配")
    except Exception as e:
        print("❌ 模型加载失败：", repr(e))
        print("   常见原因：ctranslate2 为 CPU 版，或驱动/CUDA/cuDNN 不匹配。")
        return

    # 若未提供音频，仅验证加载与设备
    if not audio:
        print("模型已成功加载。若需端到端验证，请提供 --audio /path/to/audio.wav")
        return

    # 端到端转写（中文快速验证）
    try:
        segments, info = model.transcribe(
            audio,
            language="zh",
            beam_size=1,
            vad_filter=True
        )
        print_kv("识别语言", getattr(info, "language", "unknown"))
        print_kv("语言概率", round(getattr(info, "language_probability", 0.0), 4))
        text = "".join(seg.text for seg in segments)
        print_kv("转写结果", text.strip())
    except Exception as e:
        print("❌ 转写失败：", repr(e))
        print("   若提示缺少 ffmpeg，请安装系统级 ffmpeg。某些容器里 faster-whisper 也可能需要。")

# ---------- 主入口 ----------
def main():
    parser = argparse.ArgumentParser(description="Whisper 环境体检脚本 (v2)")
    parser.add_argument("--run-inference", action="store_true",
                        help="运行一次最小推理（会下载模型）")
    parser.add_argument("--model", type=str, default="tiny",
                        help="faster-whisper 模型名（tiny/base/small/medium/large-v3 或本地 CT2 目录）")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备偏好：auto/cuda/cpu")
    parser.add_argument("--compute-type", type=str, default="float16",
                        help="计算类型：float32/float16/int8_float16/int8 等")
    parser.add_argument("--audio", type=str, default=None,
                        help="用于端到端验证的音频文件路径（可选）")

    args = parser.parse_args()

    diagnose_system()
    diagnose_torch()
    diagnose_ctranslate2_and_fw()
    diagnose_openai_and_whisper()

    if args.run_inference:
        run_minimal_inference(
            model_name=args.model,
            device_pref=args.device,
            compute_type=args.compute_type,
            audio=args.audio,
        )
    else:
        section("提示")
        print("未开启 --run-inference。若要验证端到端 GPU 推理，请加上该参数，并可指定 --model 与 --audio。")

if __name__ == "__main__":
    main()
