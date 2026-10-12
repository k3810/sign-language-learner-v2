#!/bin/bash
# 1. 혹시 켜져 있을지 모르는 서버 종료
sudo fuser -k -9 5000/tcp

# 2. 가상환경 활성화 및 서버 백그라운드 실행
source ~/hailo-rpi5-examples/venv_hailo_rpi_examples/bin/activate
python ~/hailo-rpi5-examples/avatar_server.py &

# 3. 서버가 켜질 때까지 3초 대기
sleep 3

# 4. 크롬 브라우저를 주소창 없는 '키오스크 모드(전체 화면)'로 강제 실행
chromium-browser --kiosk http://127.0.0.1:5000
