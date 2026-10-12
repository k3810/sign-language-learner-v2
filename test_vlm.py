import glob
from hailo_platform import VDevice
from hailo_platform.genai import VLM

hef_files = glob.glob('/home/a/Sign_Kiosk/models/**/*.hef', recursive=True)
model_file = hef_files[0]

print(f"[*] 단독 격리 테스트 시작: {model_file}")
try:
    with VDevice() as vdev:
        print("[*] VDevice 할당 성공. VLM 모델을 NPU에 강제 적재 중입니다...")
        vlm = VLM(model_path=model_file, vdevice=vdev)
        print("✅ VLM 모델 적재 대성공! 메모리 한계가 완벽히 해결되었습니다.")
except Exception as e:
    print(f"❌ NPU 적재 에러 상세 내용: {e}")
