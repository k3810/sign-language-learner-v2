import cv2
import os
import json
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

base_path = "/home/a/Sign_Kiosk/media/수어 영상 mp4"
words = ["감사합니다", "괜찮다"]
expert_data = {}

print("🚀 [최종 교정] 노이즈 없는 핵심 2D 궤적 추출을 시작합니다...")

for word in words:
    video_path = os.path.join(base_path, f"{word}.mp4")
    if not os.path.exists(video_path): continue
        
    cap = cv2.VideoCapture(video_path)
    sequence_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 쓰레기값인 Z축을 완전히 배제하고, 가장 정밀한 X, Y 좌표만 추출
            nose = np.array([landmarks[0].x, landmarks[0].y])
            l_shoulder = np.array([landmarks[11].x, landmarks[11].y])
            r_shoulder = np.array([landmarks[12].x, landmarks[12].y])
            l_wrist = np.array([landmarks[15].x, landmarks[15].y])
            r_wrist = np.array([landmarks[16].x, landmarks[16].y])
            
            # 어깨 중앙 기준점 및 어깨 너비 추출 (이동/원근감 완벽 보정)
            shoulder_center = (l_shoulder + r_shoulder) / 2.0
            shoulder_width = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
            
            norm_nose = (nose - shoulder_center) / shoulder_width
            norm_l_wrist = (l_wrist - shoulder_center) / shoulder_width
            norm_r_wrist = (r_wrist - shoulder_center) / shoulder_width
            
            # 6차원(X,Y * 3포인트) 벡터 완성
            frame_data = np.concatenate([norm_nose, norm_l_wrist, norm_r_wrist]).tolist()
            sequence_data.append(frame_data)
    cap.release()
    
    if sequence_data:
        expert_data[word] = sequence_data[::2] 
        print(f"✅ '{word}' 6D 궤적 추출 완료 (총 {len(expert_data[word])} 프레임)")

output_json = "/home/a/Sign_Kiosk/dynamic_sign_model.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(expert_data, f, ensure_ascii=False, indent=4)
print(f"🎉 2D 정답지 생성 완료! 저장 위치: {output_json}")
