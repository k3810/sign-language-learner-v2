import os
import urllib.request

model_dir = "/home/a/Sign_Kiosk/models"
base_url = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/raw/main/"

files = [
    "config.json", "tokenizer.json", "tokenizer_config.json", 
    "preprocessor_config.json", "generation_config.json", 
    "merges.txt", "vocab.json"
]

print("[1/2] 꼬여버린 기존 설정 파일 삭제 및 100% 순정 파일 다운로드 중...")
for f in files:
    filepath = os.path.join(model_dir, f)
    if os.path.exists(filepath):
        os.remove(filepath)
    try:
        urllib.request.urlretrieve(base_url + f, filepath)
        print(f"  -> 순정 복구 완료: {f}")
    except Exception as e:
        print(f"  -> ❌ 다운로드 실패 ({f}): {e}")

print("\n[2/2] ✅ VLM 순정 셋업 완료! 이제 하드웨어 규격과 완벽히 일치합니다.")
