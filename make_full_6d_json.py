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

print("🚀 전체 수어 영상 대상 6D 정답지 일괄 추출을 시작합니다...")

# 폴더 내의 모든 mp4 파일을 자동으로 찾아냅니다.
video_files = glob.glob(os.path.join(base_path, "*.mp4"))

for video_path in video_files:
    # 파일명에서 확장자를 제외한 단어만 추출
    word = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    sequence_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            nose = np.array([landmarks[0].x, landmarks[0].y])
            l_shoulder = np.array([landmarks[11].x, landmarks[11].y])
            r_shoulder = np.array([landmarks[12].x, landmarks[12].y])
            l_wrist = np.array([landmarks[15].x, landmarks[15].y])
            r_wrist = np.array([landmarks[16].x, landmarks[16].y])
            
            shoulder_center = (l_shoulder + r_shoulder) / 2.0
            shoulder_width = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
            
            norm_nose = (nose - shoulder_center) / shoulder_width
            norm_l_wrist = (l_wrist - shoulder_center) / shoulder_width
            norm_r_wrist = (r_wrist - shoulder_center) / shoulder_width
            
            frame_data = np.concatenate([norm_nose, norm_l_wrist, norm_r_wrist]).tolist()
            sequence_data.append(frame_data)
    cap.release()
    
    if sequence_data:
        expert_data[word] = sequence_data[::2] 
        print(f"✅ '{word}' 6D 궤적 추출 완료 (총 {len(expert_data[word])} 프레임)")

output_json = "/home/a/Sign_Kiosk/dynamic_sign_model.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(expert_data, f, ensure_ascii=False, indent=4)
print(f"🎉 전체 단어 6D 정답지 생성 완료! 저장 위치: {output_json}")
