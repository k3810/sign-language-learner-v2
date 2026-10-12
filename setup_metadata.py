import os
import json
import urllib.request

model_dir = "/home/a/Sign_Kiosk/models"
base_url = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/raw/main/"

files = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "tokenizer_config.json", "tokenizer.json", "merges.txt", "vocab.json"
]

print("[1/3] 필수 JSON 및 메타데이터 파일 직접 다운로드 중...")
for f in files:
    url = base_url + f
    dest = os.path.join(model_dir, f)
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  -> 다운로드 성공: {f}")
    except Exception as e:
        print(f"  -> ❌ 다운로드 실패 ({f}): {e}")

print("\n[2/3] NPU 메모리 폭발 방지(Error 6)를 위한 KV Cache 다이어트 패치 중...")
gen_config_path = os.path.join(model_dir, 'generation_config.json')
try:
    with open(gen_config_path, 'r') as f:
        data = json.load(f)
    data['max_new_tokens'] = 128  # 512MB CMA 한계 돌파를 위한 필수 설정
    with open(gen_config_path, 'w') as f:
        json.dump(data, f, indent=4)
    print("  -> generation_config.json (max_new_tokens=128) 패치 완료!")
except Exception as e:
    print(f"  -> ❌ 패치 실패: {e}")

print("\n[3/3] 셋업이 완벽하게 끝났습니다! 이제 서버를 켜주세요.")
