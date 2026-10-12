import cv2
import json
import time
import numpy as np
import mediapipe as mp
import os

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

json_path = "/home/a/Sign_Kiosk/dynamic_sign_model.json"

# 기존 정답지가 있으면 불러오고, 없으면 새로 만듭니다.
if os.path.exists(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        expert_data = json.load(f)
else:
    expert_data = {}

def get_3d_direction(p1, p2):
    v = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm > 0 else [0.0, 0.0, 0.0]

def extract_feature_vector(results):
    vec = []
    if results.pose_landmarks:
        lms = results.pose_landmarks.landmark
        vec.extend(get_3d_direction(lms[0], lms[13]))
        vec.extend(get_3d_direction(lms[0], lms[14]))
        vec.extend(get_3d_direction(lms[13], lms[15]))
        vec.extend(get_3d_direction(lms[14], lms[16]))
    else: vec.extend([0.0] * 12)
        
    if results.left_hand_landmarks:
        lms = results.left_hand_landmarks.landmark
        for tip_idx in [4, 8, 12, 16, 20]: vec.extend(get_3d_direction(lms[0], lms[tip_idx]))
    else: vec.extend([0.0] * 15)
        
    if results.right_hand_landmarks:
        lms = results.right_hand_landmarks.landmark
        for tip_idx in [4, 8, 12, 16, 20]: vec.extend(get_3d_direction(lms[0], lms[tip_idx]))
    else: vec.extend([0.0] * 15)
        
    arr = np.array(vec)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 0 else [0.0] * 42

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n==================================================")
print("🎥 실시간 무결점 3D 정답지 생성기 가동")
print("==================================================\n")

while True:
    word = input("👉 녹화할 수어 단어를 입력하세요 (종료하려면 'q' 입력): ").strip()
    if word.lower() == 'q': break
    if not word: continue

    print(f"\n⚠️ 카메라 정중앙에 서서 양손이 화면에 들어오는지 확인하십시오.")
    for i in range(3, 0, -1):
        print(f"⏳ {i}초 후 '{word}' 녹화를 시작합니다...")
        time.sleep(1)
        
    print("🔴 [녹화 시작] 동작을 취해주세요! (2초간 진행)")
    
    start_time = time.time()
    sequence_data = []
    
    while time.time() - start_time < 2.0:
        ret, frame = cap.read()
        if not ret: continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        # 화면 피드백
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        cv2.putText(frame, f"Recording: {word}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Ground Truth Recorder", frame)
        cv2.waitKey(1)
        
        feature = extract_feature_vector(results)
        sequence_data.append(feature)
        
    expert_data[word] = sequence_data[::2] # 다운샘플링 최적화
    print(f"✅ '{word}' 무결점 3D 벡터가 성공적으로 추출되었습니다!\n")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(expert_data, f, ensure_ascii=False, indent=4)

cap.release()
cv2.destroyAllWindows()
print("🎉 모든 정답지 갱신이 완료되었습니다. 메인 UI를 다시 실행해 보세요!")
