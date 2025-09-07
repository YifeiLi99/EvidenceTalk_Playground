import re
def redact(text: str) -> str:
    text = re.sub(r'\b1[3-9]\d{9}\b', '***PHONE***', text)    # 手机
    text = re.sub(r'[\w\.-]+@[\w\.-]+', '***EMAIL***', text)  # 邮箱
    return text
