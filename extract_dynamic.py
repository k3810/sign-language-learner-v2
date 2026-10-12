import cv2
import mediapipe as mp
import numpy as np
import json
import os

# 라즈베리파이 환경 최적화
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 무거운 Holistic을 제거하고 가벼운 Pose 전용 모듈로 교체
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

words = ["감사합니다", "괜찮다"]
expert_data = {}
base_media_path = "/home/a/Sign_Kiosk/media/수어 영상 mp4"

for word in words:
    video_path = f"{base_media_path}/{word}.mp4"
    if not os.path.exists(video_path):
        video_path = f"{word}.mp4"
        if not os.path.exists(video_path):
            print(f"⚠️ '{word}.mp4' 영상을 찾을 수 없습니다. 경로를 확인해주세요.")
            continue

    cap = cv2.VideoCapture(video_path)
    sequence_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 핵심 관절 추출: 코, 좌/우 어깨, 좌/우 손목
            nose = np.array([landmarks[0].x, landmarks[0].y])
            l_shoulder = np.array([landmarks[11].x, landmarks[11].y])
            r_shoulder = np.array([landmarks[12].x, landmarks[12].y])
            l_wrist = np.array([landmarks[15].x, landmarks[15].y])
            r_wrist = np.array([landmarks[16].x, landmarks[16].y])
            
            # 기준점(어깨 중앙) 및 스케일(어깨 너비) 계산
            shoulder_center = (l_shoulder + r_shoulder) / 2.0
            shoulder_width = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
            
            # 공간 정규화 수행 (어떤 화각이나 체형에서도 동일한 데이터 유지)
            norm_l_wrist = (l_wrist - shoulder_center) / shoulder_width
            norm_r_wrist = (r_wrist - shoulder_center) / shoulder_width
            norm_nose = (nose - shoulder_center) / shoulder_width
            
            # 1프레임의 관절 궤적 데이터 저장 (총 6개 좌표 축)
            frame_data = norm_l_wrist.tolist() + norm_r_wrist.tolist() + norm_nose.tolist()
            sequence_data.append(frame_data)
            
    cap.release()
    
    if sequence_data:
        expert_data[word] = sequence_data
        print(f"✅ '{word}' 시계열 동적 궤적 추출 완료 (총 {len(sequence_data)} 프레임)")

# 새로운 시계열 전용 JSON 파일 생성
with open("/home/a/Sign_Kiosk/dynamic_sign_model.json", "w", encoding="utf-8") as f:
    json.dump(expert_data, f, ensure_ascii=False, indent=4)
print("🚀 [dynamic_sign_model.json] 동적 마스터 데이터 파일 생성 완료!")
