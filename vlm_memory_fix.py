import os
import json
import urllib.request

model_dir = "/home/a/Sign_Kiosk/models"
base_url = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/raw/main/"

files = [
    "config.json", "tokenizer.json", "tokenizer_config.json", 
    "preprocessor_config.json", "generation_config.json", 
    "merges.txt", "vocab.json"
]

print("[1/3] VLM 구동을 위한 필수 설계도 7종 다운로드 중...")
for f in files:
    try:
        urllib.request.urlretrieve(base_url + f, os.path.join(model_dir, f))
        print(f"  -> 다운로드 완료: {f}")
    except Exception as e:
        print(f"  -> 다운로드 실패 ({f}): {e}")

print("\n[2/3] 라즈베리파이 512MB CMA를 위한 VLM 메모리 클램프(Clamp) 적용 중...")

# 1. config.json 메모리 제한 (에러 6번 해결의 핵심)
cfg_path = os.path.join(model_dir, "config.json")
try:
    with open(cfg_path, 'r') as f: cfg = json.load(f)
    cfg['max_position_embeddings'] = 1024  # 32768 -> 1024로 대폭 축소
    cfg['sliding_window'] = 1024
    with open(cfg_path, 'w') as f: json.dump(cfg, f, indent=4)
    print("  -> config.json (KV Cache 1024 락) 패치 완료!")
except Exception as e:
    print(f"  -> 패치 실패: {e}")

# 2. generation_config.json 패치 (출력 제한)
gen_path = os.path.join(model_dir, "generation_config.json")
try:
    with open(gen_path, 'r') as f: gen = json.load(f)
    gen['max_new_tokens'] = 128
    with open(gen_path, 'w') as f: json.dump(gen, f, indent=4)
    print("  -> generation_config.json (max_new_tokens=128) 패치 완료!")
except Exception as e:
    print(f"  -> 패치 실패: {e}")

print("\n[3/3] ✅ VLM 메모리 최적화 셋업이 완료되었습니다. 서버를 켜주세요.")
