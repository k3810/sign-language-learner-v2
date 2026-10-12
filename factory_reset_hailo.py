import os
import json
import urllib.request

model_dir = "/home/a/Sign_Kiosk/models"

print("[1/3] 기존 찌꺼기 파일 청소 중...")
for f in os.listdir(model_dir):
    if f.endswith('.json') or f.endswith('.txt'):
        os.remove(os.path.join(model_dir, f))

print("[2/3] 필수 메타데이터 다운로드 중...")
base_url = "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/raw/main/"
files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
for f in files:
    urllib.request.urlretrieve(base_url + f, os.path.join(model_dir, f))

print("[3/3] NPU 메모리 폭발(Error 6) 방지를 위한 강제 다이어트 패치...")
cfg_path = os.path.join(model_dir, "config.json")
with open(cfg_path, 'r') as f:
    cfg = json.load(f)

# 서버용 설정을 라즈베리파이/Hailo 한계치로 강제 락(Lock)
cfg['max_position_embeddings'] = 1024
cfg['max_window_size'] = 1024
cfg['use_sliding_window'] = False

with open(cfg_path, 'w') as f:
    json.dump(cfg, f, indent=4)

print("✅ RPi5 Hailo-10H 전용 최적화 셋업이 완벽하게 끝났습니다!")
