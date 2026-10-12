import cv2
import mediapipe as mp
import numpy as np
import json
import os

# 라즈베리파이 환경 최적화
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5)

# 동영상 경로 자동 탐색 설정
words = ["감사합니다", "괜찮다"]
expert_data = {}
base_media_path = "/home/a/Sign_Kiosk/media/수어 영상 mp4"

for word in words:
    video_path = f"{base_media_path}/{word}.mp4"
    if not os.path.exists(video_path):
        # 현재 폴더에 있는지도 확인
        video_path = f"{word}.mp4"
        if not os.path.exists(video_path):
            print(f"⚠️ '{word}.mp4' 영상을 찾을 수 없습니다. 경로를 확인해주세요.")
            continue

    cap = cv2.VideoCapture(video_path)
    best_frame_landmarks = None
    max_extension = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand = results.right_hand_landmarks or results.left_hand_landmarks
        
        if hand:
            pts = np.array([[lm.x, lm.y] for lm in hand.landmark])
            wrist = pts[0]
            base = pts[9]
            
            # 공간 정규화 스케일 계산
            scale = np.linalg.norm(base - wrist) + 1e-6
            extensions = [np.linalg.norm(pts[i] - wrist) / scale for i in [8, 12, 16, 20]]
            avg_ext = np.mean(extensions)
            
            if avg_ext > max_extension:
                max_extension = avg_ext
                normalized_pts = (pts - wrist) / scale
                best_frame_landmarks = normalized_pts.tolist()
    
    if best_frame_landmarks:
        expert_data[word] = best_frame_landmarks
        print(f"✅ '{word}' 핵심 포인트 추출 및 정규화 완료!")
    cap.release()

with open("/home/a/Sign_Kiosk/open_sign_model_v1.json", "w", encoding="utf-8") as f:
    json.dump(expert_data, f, ensure_ascii=False, indent=4)
print("🚀 [open_sign_model_v1.json] 라즈베리파이 로컬 생성 완료!")
