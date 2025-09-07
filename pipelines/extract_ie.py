import json
from jsonschema import validate
from openai import OpenAI

def extract_profile(turns, schema, system_prompt, model, base_url=None):
    client = OpenAI(base_url=base_url) if base_url else OpenAI()
    user_text = "\n".join(turns)
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},  # 让它只吐 JSON
        temperature=0.2,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":f"以下是客户发言（带行号），请按 schema 输出JSON：\n{user_text}"}
        ]
    )
    js = json.loads(resp.choices[0].message.content)
    validate(js, schema)  # 校验失败会抛错
    return js
