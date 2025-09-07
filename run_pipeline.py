import os, json, sys
from configs import config
from pipelines.asr import transcribe
from pipelines.redact import redact
from pipelines.extract_ie import extract_profile

def main(audio_path):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1) 转写
    turns = transcribe(audio_path, config.ASR_MODEL, config.ASR_LANG)

    # 2) 脱敏 + 粗暴限长（防上下文过长）
    turns = [redact(t) for t in turns]
    total_chars = sum(len(t) for t in turns)
    if total_chars > config.MAX_CHARS_TO_LLM:
        turns = turns[: max(10, int(len(turns)*0.6))]

    # 3) 读 schema & 提示词
    with open("schemas/customer_profile.schema.json","r",encoding="utf-8") as f:
        schema = json.load(f)
    with open("prompts/extract_profile_system.txt","r",encoding="utf-8") as f:
        sys_prompt = f.read()

    # 4) 一次调用 LLM 抽取
    profile = extract_profile(
        turns=turns,
        schema=schema,
        system_prompt=sys_prompt,
        model=config.OPENAI_MODEL,
        base_url=config.OPENAI_BASE_URL
    )

    # 5) 保存
    base = os.path.splitext(os.path.basename(audio_path))[0]
    out_path = os.path.join(config.OUTPUT_DIR, f"{base}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    print(f"✅ Done: {out_path}")

if __name__ == "__main__":
    main(sys.argv[1])
