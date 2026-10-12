import re
import os

filepath = '/home/a/Sign_Kiosk/vlm_server.py'
if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # VDevice 임포트 추가
    if 'from hailo_platform import VDevice' not in content:
        content = 'from hailo_platform import VDevice\n' + content

    # vdevice 파라미터 강제 주입
    content = re.sub(
        r'vlm\s*=\s*VLM\(model_path=model_path\)', 
        'target_vdevice = VDevice()\nvlm = VLM(model_path=model_path, vdevice=target_vdevice)', 
        content
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("[시스템] vlm_server.py 에러 수정 및 패치 완료")
else:
    print("[오류] vlm_server.py 파일을 찾을 수 없습니다.")
