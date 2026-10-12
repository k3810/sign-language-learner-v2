import cv2
import os
import json
import numpy as np
import mediapipe as mp
import glob

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

base_path = "/home/a/Sign_Kiosk/media/수어 영상 mp4"
expert_data = {}

print("🚀 [3D 공간 벡터] Z축 깊이 및 독립 관절 각도 정답지 추출 시작...", flush=True)

# 💡 핵심: 두 점 사이의 3D 방향(각도) 벡터를 구하는 함수
def get_3d_direction(p1, p2):
    v = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z]) # Z축(깊이) 추가!
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm > 0 else [0.0, 0.0, 0.0]

video_files = glob.glob(os.path.join(base_path, "*.mp4"))
for video_path in video_files:
    word = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    sequence_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            
            # 7개의 핵심 뼈대 방향을 3D로 독립 추출 (크기 무관, 오직 각도와 방향만)
            b1 = get_3d_direction(lms[0], lms[13]) # 코 -> 왼팔꿈치
            b2 = get_3d_direction(lms[0], lms[14]) # 코 -> 오른팔꿈치
            b3 = get_3d_direction(lms[13], lms[15]) # 왼팔꿈치 -> 왼손목
            b4 = get_3d_direction(lms[14], lms[16]) # 오른팔꿈치 -> 오른손목
            b5 = get_3d_direction(lms[15], lms[19]) # 왼손목 -> 왼검지
            b6 = get_3d_direction(lms[16], lms[20]) # 오른손목 -> 오른검지
            b7 = get_3d_direction(lms[15], lms[16]) # 양손목 사이의 상대 방향
            
            # 21차원 3D 벡터 조립
            vec = np.array(b1 + b2 + b3 + b4 + b5 + b6 + b7)
            norm = np.linalg.norm(vec)
            if norm > 0: vec = vec / norm
            
            sequence_data.append(vec.tolist())
    cap.release()
    
    if sequence_data:
        expert_data[word] = sequence_data[::2] 
        print(f"✅ '{word}' 3D 벡터 추출 완료", flush=True)

output_json = "/home/a/Sign_Kiosk/dynamic_sign_model.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(expert_data, f, ensure_ascii=False, indent=4)
print(f"🎉 3D 무적 정답지 생성 완료!", flush=True)
