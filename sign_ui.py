import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "2"

import sys
import shutil
import cv2
import csv
import random
import time
import threading
import json
import queue  
import traceback
import unicodedata
from datetime import datetime
from collections import deque
import numpy as np
import mediapipe as mp

try:
    from PIL import Image as PILImage
    PILLOW_READY = True
except ImportError:
    PILLOW_READY = False

try:
    from gtts import gTTS
    TTS_READY = True
except ImportError:
    TTS_READY = False

mp_holistic = mp.solutions.holistic

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTabWidget, QLabel, 
                             QListWidget, QProgressBar, QStackedWidget, QSizePolicy, QMessageBox, QDialog, QGridLayout)
from PyQt5.QtCore import Qt, QUrl, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QPixmap, QImage

import google.generativeai as genai

try:
    api_key = "AQ.Ab8RN6J3ygX4hk46pGZAKP5ZZ43Za79VO5tESQM_eov_cCyO4g"
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    GEMINI_READY = True
    print("🟢 [시스템] 구글 Gemini AI 모델 초기화 성공", flush=True)
except Exception as e:
    GEMINI_READY = False
    print(f"🔴 [시스템] 구글 Gemini AI 모델 초기화 실패: {e}", flush=True)

def get_chosung(text):
    chosung_list = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    result = ""
    for char in text:
        if '가' <= char <= '힣':
            char_code = ord(char) - 0xAC00
            chosung_index = char_code // 588
            result += chosung_list[chosung_index]
        else: result += char
    return result

class QuizResultDialog(QDialog):
    def __init__(self, correct_count, total_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle("퀴즈 결과 분석")
        self.setFixedSize(500, 350)
        self.setStyleSheet("background-color: #2b2b2b; border-radius: 15px; font-family: 'NanumGothic';")
        layout = QVBoxLayout()
        lbl_title = QLabel("최종 성적표")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("color: #ffb74d; font-size: 32px; font-weight: bold; margin-top: 10px;")
        
        score_percent = int((correct_count / total_count) * 100) if total_count > 0 else 0
        lbl_score = QLabel(f"{score_percent} 점" if total_count != float('inf') else f"{correct_count}문제 연속 성공!")
        lbl_score.setAlignment(Qt.AlignCenter)
        
        if total_count == float('inf'): color, msg = "#FFD700", "서바이벌 종료! 대단한 집중력입니다!"
        else:
            if score_percent >= 80: color, msg = "#4CAF50", "최고의 실력입니다! 완벽해요!"
            elif score_percent >= 40: color, msg = "#FFD700", "조금만 더 연습하면 완벽해질 거예요!"
            else: color, msg = "#f44336", "학습 모드에서 다시 연습해 봅시다!"
            
        lbl_score.setStyleSheet(f"color: {color}; font-size: 70px; font-weight: bold; margin: 10px 0px;")
        lbl_detail = QLabel(f"총 {total_count if total_count != float('inf') else '무한'}문제 중 {correct_count}문제를 통과했습니다.\n\n{msg}")
        lbl_detail.setAlignment(Qt.AlignCenter)
        lbl_detail.setStyleSheet("color: white; font-size: 20px;")
        
        btn_close = QPushButton("확인 및 닫기")
        btn_close.setStyleSheet("background-color: #008CBA; color: white; font-size: 20px; font-weight: bold; padding: 15px; border-radius: 10px;")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_score)
        layout.addWidget(lbl_detail)
        layout.addWidget(btn_close)
        self.setLayout(layout)

class SignLanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("수어 소통 및 학습 플랫폼 PRO (75차원 완벽 엔진)")
        self.resize(1024, 600)
        self.showFullScreen()
        
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.base_path = "/home/a/Sign_Kiosk/media/수어 영상 mp4"
        self.default_expert_img_path = "/home/a/Sign_Kiosk/media/전문가 시연 이미지.png"
        self.json_model_path = "/home/a/Sign_Kiosk/dynamic_sign_model.json"
        
        self.app_started = False  
        self.initial_load = True
        self.is_recording = False
        self.is_counting_down = False
        self.countdown_start_time = 0
        self.record_start_time = 0
        self.analysis_mode = ""
        self.current_target_word = ""
        self.is_auto_running = False 
        self.vision_buffer = [] 

        self.quiz_type = ""
        self.quiz_current_q = 0
        self.quiz_correct_count = 0
        self.quiz_target_word = ""
        
        self.is_processing_frame = False
        self.pose_buffer = deque(maxlen=100)
        self.frame_count = 0
        self.ui_queue = queue.Queue()
        
        self.last_pose = [0.0]*12
        self.last_lh = [0.0]*15
        self.last_rh = [0.0]*15
        self.last_rel = [0.0]*9
        self.last_rel_head = [0.0]*6
        self.last_torso = [0.0]*18

        self.word_data = {
            "감사합니다": "손끝이 밖으로 향하게 펴서 모로 세운 오른손의 4지 옆면을 손바닥이 아래로 향하게 편 왼 손등에 두 번 댄다.",
            "괜찮다": "오른 주먹의 4지를 펴서 끝 바닥을 턱에 가볍게 두 번 댄다.",
            "안녕하세요": "오른 손바닥으로 주먹을 쥔 왼 팔을 쓸어내린 다음, 두 주먹을 쥐고 바닥이 아래로 향하게하여 가슴 앞에서 아래로 내린다.",
            "못생기다": "얼굴 앞에서 오므렸던 오른손을 얼굴 쪽으로 끌어들이며 편다.",
            "미안합니다": "오른손의 1·5지 끝을 맞대어 동그라미를 만들어 이마에 댔다가 1·5지를 펴며 내린다.",
            "사랑합니다": "5지 바닥을 1지 옆면에 대고 손등이 왼쪽으로 향하게 모로 세운 왼 주먹 위에 오른 손바닥을 대고 오른손만 오른쪽으로 돌린다.",
            "정말로": "5지를 접고 나머지 손가락을 펴서 손바닥이 왼쪽으로 향하게 세운 오른손의 1지 옆면을 턱 중앙에 댄다.",
            "기다리다": "오른손을 ‘ㄱ’자 모양으로 구부려 끝이 약간 안으로 향하게 하여 손가락 등을 턱 밑에 댄다.",
            "좋다": "오른 주먹을 코에 1·5지 옆면이 닿게 댄다.",
            "만나다": "두 주먹의 1지를 펴서 마주 세웠다가 중앙으로 모아 마주 댄다.",
            "아니다": "두 주먹의 1·5지를 펴서 끝이 마주 보게 하였다가 손목을 양옆으로 돌린다.",
            "그": "왼 주먹의 5지를 펴서 바닥이 밖으로 향하게 세우고, 그 등을 오른 주먹의 1지를 펴서 끝으로 가리킨다",
            "기역": "오른 주먹의 1·5지를 펴서 1지 끝이 아래로 손등이 밖으로 향하게 세운다.",
            "나": "오른 손바닥을 가슴 중앙에 댄다.",
            "나이": "오른 손등을 턱에 대고 손가락을 자유로이 상하로 움직인다.",
            "남자": "두 주먹의 5지를 펴서 세워 5지 옆면을 두 번 댔다 뗀다.",
            "너": "오른 주먹의 1지를 펴서 끝이 밖으로 향하게 하여 약간 내민다.",
            "넷": "5지를 접고 1·2·3·4지를 벌려 바닥이 밖으로 향하게 세운다.",
            "니은": "오른 주먹의 1·5지를 펴서 1지 끝이 왼쪽으로 손등이 밖으로 향하게 한다.",
            "다섯": "손등이 안으로 향하게 쥔 오른 주먹의 5지를 펴서 눕힌다.",
            "돕다": "왼 주먹의 5지를 펴서 바닥이 밖으로 향하게 세우고, 그 등에 오른 손바닥을 세워 두 번 댄다.",
            "둘": "오른 주먹의 1·2지를 펴서 벌려 바닥이 밖으로 향하게 세운다.",
            "디귿": "오른 주먹의 1·2지를 펴서 벌려 끝이 왼쪽으로 손등이 밖으로 향하게 한다.",
            "리을": "오른 주먹의 1·2·3지를 펴서 벌려 끝이 왼쪽으로 손등이 밖으로 향하게 한다.",
            "산": "오른 주먹의 2지를 펴서 등이 밖으로 향하게 세운다.",
            "셋": "오른 주먹의 1·2·3지를 펴서 바닥이 밖으로 향하게 세운다.",
            "시옷": "오른 주먹의 1·2지를 펴서 벌려 등이 밖으로 끝이 아래로 향하게 한다.",
            "십": "오른 주먹의 1지를 펴서 약간 구부려 끝이 밖으로 향하게 하여 좌우로 약간 흔든다.",
            "아홉": "오른손을 펴서 손바닥이 안으로 향하게 비스듬히 세운다.",
            "여기": "오른 주먹의 1지를 펴서 끝이 약간 아래로 향하게 하여 약간 내린다.",
            "여덟": "오른손을, 4지를 접고 나머지 손가락을 벌려 손등이 밖으로 향하게 비스듬히 세운다.",
            "여섯": "오른 주먹의 1·5지를 펴서 등이 밖으로 향하게 모로 세운다.",
            "여자": "오른 주먹의 4지를 펴서 등이 밖으로 향하게 세운다.",
            "열다섯": "오른 주먹의 1지를 펴서 바닥이 밖으로 향하게 세워 굽히며 5지를 편다.",
            "열여덟": "오른손의 1지를 굽히고 2·3·5지를 펴서 등이 밖으로 향하게 모로 세운다.",
            "이응": "손바닥이 밖으로 향하게 펴서 세운 오른손의 1·5지 끝을 맞대어 동그라미를 만들어 보인다.",
            "일등": "오른 주먹의 1지를 펴서 왼쪽 어깨에 댄다.",
            "지읒": "오른 주먹의 1·2·5지를 펴서 등이 밖으로 끝이 아래로 향하게 세운다.",
            "책": "두 손을 펴서 손끝이 밖으로 향하게 하여 맞댔다가 손목을 양옆으로 돌려 편다.",
            "치읓": "오른 주먹의 1·2·3·5지를 펴서 등이 밖으로 끝이 아래로 향하게 세운다.",
            "키읔": "오른 주먹의 1지를 반쯤 굽히고 2·5지를 펴서 등이 밖으로, 끝이 아래로 향하게 세운다.",
            "티읕": "손등이 밖으로 향하게 쥔 오른 주먹의 1·2·3지를 펴서 끝이 왼쪽으로 향하게 하여 2·3지는 붙이고 1지는 뗀다.",
            "피읖": "손등이 안으로 향하게 세운 오른손의 1·2·3·4·5지를 반쯤 굽힌다.",
            "필요": "두 손을 구부려 손가락 등이 아래로 향하게 하고 끝으로 배 양쪽을 두 번 두드린다.",
            "하나": "오른 주먹의 1지를 펴서 바닥이 밖으로 향하게 세운다.",
            "핸드폰": "1지를 펴서 세운 오른 주먹의 바닥을 오른쪽 귀에 댄다.",
            "히읗": "오른 주먹의 5지를 펴서 바닥이 밖으로 향하게 세운다."
        }

        self.expert_data = {}
        self.words = list(self.word_data.keys())
        if os.path.exists(self.json_model_path):
            with open(self.json_model_path, 'r', encoding='utf-8') as f:
                raw_json = json.load(f)
                for k, v in raw_json.items():
                    res = self.resample_sequence(v, 30)
                    if res is not None:
                        self.expert_data[k] = res.tolist()
            print(f"🟢 [시스템] 정답지 데이터 로드 완료 (단어 수: {len(self.expert_data)})", flush=True)

        self.root_stack = QStackedWidget()
        self.setCentralWidget(self.root_stack)

        self.intro_widget = QWidget()
        self.intro_widget.setObjectName("IntroWidget")
        self.intro_widget.setStyleSheet("""
            #IntroWidget {
                border-image: url(/usr/share/rpd-wallpaper/aurora.jpg) 0 0 0 0 stretch stretch;
            }
            QLabel { font-family: 'NanumGothic'; }
        """)
        intro_layout = QVBoxLayout(self.intro_widget)
        
        lbl_main_title = QLabel("수어(수화) 배우기")
        lbl_main_title.setAlignment(Qt.AlignCenter)
        lbl_main_title.setStyleSheet("font-size: 60px; font-weight: bold; color: white; background-color: rgba(0,0,0,150); padding: 20px; border-radius: 15px; margin-bottom: 50px;")
        
        btn_enter = QPushButton("🚀 시작하기")
        btn_enter.setStyleSheet("background-color: #008CBA; color: white; font-size: 30px; font-weight: bold; padding: 30px; border-radius: 15px; margin: 20px 200px;")
        btn_enter.clicked.connect(self.enter_main_app)
        
        btn_intro_exit = QPushButton("❌ 종료하기")
        btn_intro_exit.setStyleSheet("background-color: #f44336; color: white; font-size: 30px; font-weight: bold; padding: 30px; border-radius: 15px; margin: 20px 200px;")
        btn_intro_exit.clicked.connect(self.close_app)
        
        intro_layout.addStretch()
        intro_layout.addWidget(lbl_main_title)
        intro_layout.addWidget(btn_enter)
        intro_layout.addWidget(btn_intro_exit)
        intro_layout.addStretch()

        self.main_app_widget = QWidget()
        self.main_app_widget.setStyleSheet("background-color: #1e1e1e; font-family: 'NanumGothic';")
        layout = QVBoxLayout(self.main_app_widget)

        top_bar = QHBoxLayout()
        self.btn_window = QPushButton("창 모드")
        self.btn_fullscreen = QPushButton("전체화면")
        self.btn_exit = QPushButton("종료하기")
        btn_style = "background-color: #008CBA; color: white; padding: 8px 15px; font-size: 14px; font-weight: bold; border-radius: 5px;"
        self.btn_window.setStyleSheet(btn_style)
        self.btn_fullscreen.setStyleSheet(btn_style)
        self.btn_exit.setStyleSheet("background-color: #f44336; color: white; padding: 8px 15px; font-size: 14px; font-weight: bold; border-radius: 5px;" )
        self.btn_window.clicked.connect(self.showNormal)
        self.btn_fullscreen.clicked.connect(self.showFullScreen)
        self.btn_exit.clicked.connect(self.close_app)
        top_bar.addStretch()
        top_bar.addWidget(self.btn_window)
        top_bar.addWidget(self.btn_fullscreen)
        top_bar.addWidget(self.btn_exit)
        layout.addLayout(top_bar)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(False)
        self.tab_learn = QWidget()
        self.tab_practice = QWidget()
        self.tab_test = QWidget()
        self.tab_auto = QWidget()
        
        self.tabs.addTab(self.tab_learn, "학습 모드")
        self.tabs.addTab(self.tab_practice, "연습 모드")
        self.tabs.addTab(self.tab_test, "확인 모드") 
        self.tabs.addTab(self.tab_auto, "자율 모드")
        self.tabs.currentChanged.connect(self.on_tab_changed)

        self.setup_learn_tab()
        self.setup_practice_tab()
        self.setup_test_tab()
        self.setup_auto_tab()
        
        layout.addWidget(self.tabs)
        self.apply_styles()
        
        self.root_stack.addWidget(self.intro_widget)
        self.root_stack.addWidget(self.main_app_widget)
        self.root_stack.setCurrentIndex(0) 

        if self.words: self.change_word_by_index(0)

        self.cap = None
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                ret, _ = cap.read()
                if ret: 
                    self.cap = cap
                    print(f"🟢 [카메라] 웹캠 초기화 성공 (인덱스: {i})", flush=True)
                    break
                else: cap.release()
            else: cap.release()
            
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)
        self.speed_timer = QTimer()
        self.speed_timer.timeout.connect(self.update_speed_timer)
        self.speed_time_left = 50 

    def close_app(self): 
        print("🛑 [시스템] 프로그램 종료 명령 호출", flush=True)
        self.close()

    def enter_main_app(self):
        print("▶️ [UI 액션] '시작하기' 버튼 클릭 - 메인 시스템 진입", flush=True)
        self.app_started = True 
        self.root_stack.setCurrentIndex(1)
        self.timer.start(35)

    def play_sound_effect(self, effect_name):
        audio_dir = os.path.join(self.base_path, "오디오 파일")
        target_audio = os.path.join(audio_dir, f"{effect_name}.mp3")
        if os.path.exists(target_audio):
            threading.Thread(target=lambda: os.system(f'mpg123 -q "{target_audio}" &'), daemon=True).start()

    def play_word_audio(self, word):
        if not self.app_started: return 
        self.stop_tts_audio()
        audio_dir = os.path.join(self.base_path, "오디오 파일")
        target_audio = None
        clean_target = unicodedata.normalize('NFC', word.strip().replace(" ", ""))
        if os.path.exists(audio_dir):
            for file in os.listdir(audio_dir):
                name, ext = os.path.splitext(file)
                clean_name = unicodedata.normalize('NFC', name.strip().replace(" ", ""))
                if clean_target == clean_name and ext.lower() == '.mp3':
                    target_audio = os.path.join(audio_dir, file)
                    break
        if target_audio:
            print(f"🔊 [오디오] 단어 원어 음성 재생: {word}", flush=True)
            threading.Thread(target=lambda: os.system(f'mpg123 -q "{target_audio}" &'), daemon=True).start()

    def stop_tts_audio(self):
        print("🛑 [오디오 제어] 재생 중인 음성 강제 종료 (Kill Switch 발동)", flush=True)
        os.system('killall -9 mpg123 2>/dev/null')
        os.system('pkill -f mpg123 2>/dev/null')

    def speak_text(self, text):
        if not TTS_READY: return
        self.stop_tts_audio() 
        def run_tts():
            try:
                print(f"💬 [TTS 엔진] 음성 합성 시작: {text[:20]}...", flush=True)
                tts = gTTS(text=text, lang='ko')
                temp_file = '/tmp/sign_chat_tts.mp3'
                tts.save(temp_file)
                os.system(f'mpg123 -q {temp_file}')
                print("🏁 [TTS 엔진] 음성 출력 완료", flush=True)
            except Exception as e: 
                print(f"❌ [TTS 오류] {e}", flush=True)
        threading.Thread(target=run_tts, daemon=True).start()

    def create_word_selection_panel(self, list_widget):
        panel = QVBoxLayout()
        lbl_category = QLabel("학습 단어 목록")
        lbl_category.setStyleSheet("font-size: 20px; font-weight: bold; color: #4CAF50;")
        list_widget.setFixedWidth(220)
        list_widget.addItems(self.words)
        list_widget.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        list_widget.itemClicked.connect(self.on_item_clicked)
        list_widget.currentRowChanged.connect(self.change_word_by_index)
        panel.addWidget(lbl_category)
        panel.addWidget(list_widget)
        return panel

    def on_tab_changed(self, index):
        tab_names = ["학습 모드", "연습 모드", "확인 모드", "자율 모드"]
        print(f"🔄 [UI 액션] 탭 이동: {tab_names[index]}", flush=True)
        self.stop_tts_audio()
        if hasattr(self, 'learn_media_player'): self.learn_media_player.pause()
        self.is_recording = False
        self.is_counting_down = False
        self.pose_buffer.clear()
        self.vision_buffer = []
        if hasattr(self, 'btn_check_motion'):
            self.btn_check_motion.setEnabled(True)
            if hasattr(self, 'practice_progress') and self.current_target_word:
                self.practice_progress.setFormat(f"[{self.current_target_word}] 동작 연습하기")
        if hasattr(self, 'btn_submit_act'):
            self.btn_submit_act.setEnabled(True)
            if hasattr(self, 'lbl_quiz_target_act') and self.quiz_target_word:
                if self.quiz_type == "action": self.lbl_quiz_target_act.setText(f"제시어: [{self.quiz_target_word}]")
                elif self.quiz_type == "initial": self.lbl_quiz_target_act.setText(f"초성: [{get_chosung(self.quiz_target_word)}]")
                elif self.quiz_type == "survival": self.lbl_quiz_target_act.setText(f"제시어: [{self.quiz_target_word}]")
                self.lbl_quiz_target_act.setStyleSheet("font-size: 36px; font-weight: bold; color: #FFD700; background-color: #2b2b2b; padding: 20px; border-radius: 15px;")
        if hasattr(self, 'btn_auto_scan'):
            self.btn_auto_scan.setEnabled(True)
            self.btn_auto_scan.setText("동작 2초 녹화 후 AI 번역")
            self.btn_auto_scan.setStyleSheet("background-color: #4CAF50; color: white; font-size: 22px; font-weight: bold; padding: 15px; border-radius: 8px;")
            if hasattr(self, 'lbl_auto_result'):
                self.lbl_auto_result.setText("준비가 되면 아래 버튼을 누르고 2초간 수어 동작을 취해보세요.")
        if index == 2: self.test_stack.setCurrentIndex(0)

    def on_item_clicked(self, item):
        self.initial_load = False 
        word = item.text()
        print(f"👉 [UI 액션] 리스트에서 단어 직접 클릭: '{word}'", flush=True)
        if self.learn_list.currentItem() and self.learn_list.currentItem().text() == word:
            if self.tabs.currentIndex() == 0: 
                self.play_word_audio(word)
                self.replay_video()
        else: self.change_word(word)

    def change_word_by_index(self, index):
        if index < 0 or index >= len(self.words): return
        self.change_word(self.words[index])

    def change_word(self, word):
        print(f"🔄 [시스템] 타겟 단어 변경 프로세스 진입: '{word}'", flush=True)
        self.stop_tts_audio() 
        self.current_target_word = word
        self.pose_buffer.clear()
        self.learn_list.blockSignals(True)
        if hasattr(self, 'practice_list'): self.practice_list.blockSignals(True)
        items = self.learn_list.findItems(word, Qt.MatchExactly)
        if items: self.learn_list.setCurrentItem(items[0])
        if hasattr(self, 'practice_list'):
            items_prac = self.practice_list.findItems(word, Qt.MatchExactly)
            if items_prac: self.practice_list.setCurrentItem(items_prac[0])
        self.learn_list.blockSignals(False)
        if hasattr(self, 'practice_list'): self.practice_list.blockSignals(False)
        desc = self.word_data.get(word, "설명이 없습니다.")
        if hasattr(self, 'lbl_learn_desc'): self.lbl_learn_desc.setText(desc)
        if hasattr(self, 'practice_progress'):
            self.practice_progress.setValue(0)
            self.practice_progress.setFormat(f"[{word}] 동작을 연습하세요")
        if hasattr(self, 'lbl_practice_desc_text'):
            self.lbl_practice_desc_text.setText(f"<span style='color:#4CAF50; font-size: 20px; font-weight: bold;'>💡 수형 설명: {desc}</span>")
            self.lbl_practice_ai_feedback.setText("AI 피드백 대기 중...")
        if not self.initial_load and self.tabs.currentIndex() == 0:
            self.play_word_audio(word)
        target_video = None
        target_img = None
        clean_target_word = unicodedata.normalize('NFC', word.strip().replace(" ", ""))
        if os.path.exists(self.base_path):
            for file in os.listdir(self.base_path):
                name, ext = os.path.splitext(file)
                clean_name = unicodedata.normalize('NFC', name.strip().replace(" ", ""))
                ext = ext.lower()
                if clean_target_word == clean_name:
                    if ext in ['.mp4', '.avi', '.mov']: target_video = os.path.join(self.base_path, file)
                    elif ext in ['.jpg', '.jpeg', '.png']: target_img = os.path.join(self.base_path, file)
        if target_img:
            safe_img_path = "/tmp/temp_sign_img.jpg"
            try:
                shutil.copy2(target_img, safe_img_path)
                pixmap = QPixmap(safe_img_path).scaled(450, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                if hasattr(self, 'lbl_learn_image'): self.lbl_learn_image.setPixmap(pixmap)
                if hasattr(self, 'lbl_practice_image'): self.lbl_practice_image.setPixmap(pixmap)
            except Exception: pass
        else:
            no_img_style = "background-color: #3a3a3a; border-radius: 10px; color: #ff4444; font-size: 16px;"
            if hasattr(self, 'lbl_learn_image'):
                self.lbl_learn_image.setText("사진 없음")
                self.lbl_learn_image.setStyleSheet(no_img_style)
            if hasattr(self, 'lbl_practice_image'):
                self.lbl_practice_image.setText("사진 없음")
                self.lbl_practice_image.setStyleSheet(no_img_style)
        self.current_video_path = target_video
        if target_video and self.tabs.currentIndex() == 0:
            if not self.initial_load: self.play_video()
        else:
            if hasattr(self, 'learn_media_player'):
                self.learn_media_player.stop()
                self.learn_video_stack.setCurrentIndex(0)

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.learn_video_stack.setCurrentIndex(0)

    def setup_learn_tab(self):
        layout = QHBoxLayout()
        self.learn_list = QListWidget()
        left_panel = self.create_word_selection_panel(self.learn_list)
        right_panel = QVBoxLayout()
        top_half = QVBoxLayout()
        lbl_video_title = QLabel("전문 시연 영상")
        lbl_video_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffb74d;")
        self.learn_video_stack = QStackedWidget()
        self.learn_default_img = QLabel()
        self.learn_default_img.setAlignment(Qt.AlignCenter)
        if os.path.exists(self.default_expert_img_path):
            self.learn_default_img.setPixmap(QPixmap(self.default_expert_img_path).scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.learn_video_widget = QVideoWidget()
        self.learn_media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.learn_media_player.setVideoOutput(self.learn_video_widget)
        self.learn_media_player.mediaStatusChanged.connect(self.on_media_status_changed)
        self.learn_video_stack.addWidget(self.learn_default_img)
        self.learn_video_stack.addWidget(self.learn_video_widget)
        self.btn_replay_learn = QPushButton("영상 다시보기")
        self.btn_replay_learn.setStyleSheet("background-color: #ff9800; color: white; font-size: 16px; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_replay_learn.clicked.connect(self.replay_video)
        top_half.addWidget(lbl_video_title)
        top_half.addWidget(self.learn_video_stack, stretch=1)
        top_half.addWidget(self.btn_replay_learn)
        bottom_half = QVBoxLayout()
        lbl_img_title = QLabel("수형 이미지 및 설명")
        lbl_img_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4CAF50;")
        self.lbl_learn_image = QLabel("사진 없음")
        self.lbl_learn_image.setStyleSheet("background-color: transparent;")
        self.lbl_learn_image.setAlignment(Qt.AlignCenter)
        self.lbl_learn_desc = QLabel("단어를 선택해 주세요.")
        self.lbl_learn_desc.setWordWrap(True)
        self.lbl_learn_desc.setStyleSheet("font-size: 20px; color: #FFD700; font-weight: bold; padding-top: 5px;")
        self.lbl_learn_desc.setAlignment(Qt.AlignCenter)
        bottom_half.addWidget(lbl_img_title)
        bottom_half.addWidget(self.lbl_learn_image, stretch=1)
        bottom_half.addWidget(self.lbl_learn_desc)
        right_panel.addLayout(top_half, 6) 
        right_panel.addLayout(bottom_half, 4)
        layout.addLayout(left_panel, 1)
        layout.addLayout(right_panel, 3)
        self.tab_learn.setLayout(layout)

    def setup_practice_tab(self):
        tab_layout = QHBoxLayout()
        self.practice_list = QListWidget()
        left_panel = self.create_word_selection_panel(self.practice_list)
        right_panel = QVBoxLayout()
        lbl_cam_title = QLabel("내 동작 분석 및 구글 AI 피드백")
        lbl_cam_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #008CBA;")
        self.lbl_practice_camera = QLabel("카메라 대기 중...")
        self.lbl_practice_camera.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.lbl_practice_camera.setStyleSheet("background-color: black; border-radius: 10px;")
        self.lbl_practice_camera.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) 
        self.practice_progress = QProgressBar()
        self.practice_progress.setRange(0, 100)
        self.practice_progress.setValue(0)
        self.practice_progress.setFormat("버튼을 누르고 동작하세요")
        self.practice_progress.setFixedHeight(35)
        self.practice_progress.setStyleSheet("""
            QProgressBar { border: 2px solid grey; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; color: white; background-color: #2b2b2b;}
            QProgressBar::chunk { background-color: #4CAF50; border-radius: 8px;}
        """)
        self.lbl_practice_desc_text = QLabel("💡 수형 설명: 단어를 선택해주세요.")
        self.lbl_practice_desc_text.setWordWrap(True)
        self.lbl_practice_desc_text.setStyleSheet("font-size: 20px; font-weight: bold; color: #4CAF50; margin: 5px 0px;")
        self.lbl_practice_ai_feedback = QLabel("AI 피드백 대기 중...")
        self.lbl_practice_ai_feedback.setWordWrap(True)
        self.lbl_practice_ai_feedback.setAlignment(Qt.AlignCenter)
        self.lbl_practice_ai_feedback.setStyleSheet("font-size: 22px; font-weight: bold; color: #FFD700; background-color: #3a3a3a; padding: 15px; border-radius: 10px;")
        
        btn_layout = QHBoxLayout()
        self.btn_check_motion = QPushButton("동작 2초 녹화 및 정확도 판독")
        self.btn_check_motion.setStyleSheet("background-color: #4CAF50; color: white; font-size: 22px; font-weight: bold; padding: 15px; border-radius: 8px;")
        self.btn_check_motion.clicked.connect(lambda: self.start_recording("practice"))
        
        self.btn_stop_audio_prac = QPushButton("⏹️ 음성 종료")
        self.btn_stop_audio_prac.setStyleSheet("background-color: #f44336; color: white; font-size: 22px; font-weight: bold; padding: 15px; border-radius: 8px;")
        self.btn_stop_audio_prac.clicked.connect(self.stop_tts_audio)
        
        btn_layout.addWidget(self.btn_check_motion, stretch=3)
        btn_layout.addWidget(self.btn_stop_audio_prac, stretch=1)

        right_panel.addWidget(lbl_cam_title, stretch=0)
        right_panel.addWidget(self.lbl_practice_camera, stretch=6) 
        right_panel.addWidget(self.practice_progress, stretch=0)
        right_panel.addWidget(self.lbl_practice_desc_text, stretch=0)
        right_panel.addWidget(self.lbl_practice_ai_feedback, stretch=3) 
        right_panel.addLayout(btn_layout, stretch=0) 
        tab_layout.addLayout(left_panel, 1) 
        tab_layout.addLayout(right_panel, 3) 
        self.tab_practice.setLayout(tab_layout)

    def setup_test_tab(self):
        self.test_stack = QStackedWidget()
        menu_widget = QWidget()
        menu_layout = QVBoxLayout(menu_widget)
        lbl_menu_title = QLabel("🎮 원하시는 퀴즈 모드를 선택하세요 🎮")
        lbl_menu_title.setStyleSheet("font-size: 32px; font-weight: bold; color: #ffb74d;")
        lbl_menu_title.setAlignment(Qt.AlignCenter)
        menu_layout.addStretch()
        menu_layout.addWidget(lbl_menu_title)
        vbox_btns = QVBoxLayout()
        vbox_btns.setAlignment(Qt.AlignCenter)
        btn_q1 = QPushButton("1. 실전 동작 퀴즈 (단어 보고 동작하기)")
        btn_q2 = QPushButton("2. 이미지 매칭 퀴즈 (이미지 보고 단어 찾기)")
        btn_q3 = QPushButton("3. 설명 매칭 퀴즈 (설명 읽고 단어 찾기)")
        btn_q4 = QPushButton("4. ⏱️ 스피드 단어 맞추기 (타임어택)")
        btn_q5 = QPushButton("5. 🔠 수어 초성 퀴즈 (초성 보고 동작하기)")
        btn_q6 = QPushButton("6. 🔄 거꾸로 퀴즈 (이미지 보고 설명 찾기)")
        btn_q7 = QPushButton("7. 💀 서바이벌 실전 (틀릴 때까지 무한 동작)")
        btn_q_style = "background-color: #008CBA; color: white; font-size: 22px; font-weight: bold; padding: 15px 30px; text-align: left; padding-left: 30px; border-radius: 10px; margin: 5px;"
        btns = [btn_q1, btn_q2, btn_q3, btn_q4, btn_q5, btn_q6, btn_q7]
        for btn in btns:
            btn.setFixedWidth(650) 
            btn.setStyleSheet(btn_q_style)
            vbox_btns.addWidget(btn)
        btn_q1.clicked.connect(lambda: self.start_quiz_session("action"))
        btn_q2.clicked.connect(lambda: self.start_quiz_session("image"))
        btn_q3.clicked.connect(lambda: self.start_quiz_session("desc"))
        btn_q4.clicked.connect(lambda: self.start_quiz_session("speed"))
        btn_q5.clicked.connect(lambda: self.start_quiz_session("initial"))
        btn_q6.clicked.connect(lambda: self.start_quiz_session("reverse"))
        btn_q7.clicked.connect(lambda: self.start_quiz_session("survival"))
        menu_layout.addLayout(vbox_btns)
        menu_layout.addStretch()
        action_widget = QWidget()
        action_layout = QVBoxLayout(action_widget)
        self.lbl_quiz_title_act = QLabel("실전 동작 퀴즈")
        self.lbl_quiz_title_act.setStyleSheet("font-size: 28px; font-weight: bold; color: #ffb74d;")
        self.lbl_quiz_title_act.setAlignment(Qt.AlignCenter)
        self.lbl_quiz_target_act = QLabel("버튼을 누르면 문제가 나옵니다")
        self.lbl_quiz_target_act.setStyleSheet("font-size: 36px; font-weight: bold; color: white; background-color: #3a3a3a; padding: 20px; border-radius: 15px;")
        self.lbl_quiz_target_act.setAlignment(Qt.AlignCenter)
        self.lbl_test_camera = QLabel("카메라 대기 중...")
        self.lbl_test_camera.setAlignment(Qt.AlignCenter)
        self.lbl_test_camera.setStyleSheet("background-color: black; border-radius: 10px;")
        btn_action_layout = QHBoxLayout()
        self.btn_submit_act = QPushButton("정답 제출 (1초 퀵 스캔)")
        self.btn_submit_act.setStyleSheet("background-color: #4CAF50; color: white; font-size: 20px; font-weight: bold; padding: 15px; border-radius: 10px;")
        self.btn_submit_act.clicked.connect(lambda: self.start_recording("quiz_act"))
        self.btn_exit_quiz1 = QPushButton("퀴즈 포기 및 나가기")
        self.btn_exit_quiz1.setStyleSheet("background-color: #f44336; color: white; font-size: 20px; font-weight: bold; padding: 15px; border-radius: 10px;")
        self.btn_exit_quiz1.clicked.connect(self.exit_mc_quiz)
        btn_action_layout.addWidget(self.btn_submit_act)
        btn_action_layout.addWidget(self.btn_exit_quiz1)
        action_layout.addWidget(self.lbl_quiz_title_act)
        action_layout.addWidget(self.lbl_quiz_target_act)
        action_layout.addWidget(self.lbl_test_camera, stretch=1)
        action_layout.addLayout(btn_action_layout)
        mc_widget = QWidget()
        mc_layout = QVBoxLayout(mc_widget)
        self.lbl_quiz_title_mc = QLabel("객관식 퀴즈")
        self.lbl_quiz_title_mc.setStyleSheet("font-size: 28px; font-weight: bold; color: #ffb74d;")
        self.lbl_quiz_title_mc.setAlignment(Qt.AlignCenter)
        self.speed_progress = QProgressBar()
        self.speed_progress.setRange(0, 50)
        self.speed_progress.setValue(50)
        self.speed_progress.setTextVisible(False)
        self.speed_progress.setFixedHeight(15)
        self.speed_progress.setStyleSheet("QProgressBar { border: none; background-color: #2b2b2b;} QProgressBar::chunk { background-color: #f44336; }")
        self.speed_progress.hide()
        self.mc_content_stack = QStackedWidget()
        self.lbl_mc_image = QLabel("이미지")
        self.lbl_mc_image.setAlignment(Qt.AlignCenter)
        self.lbl_mc_image.setStyleSheet("background-color: #2b2b2b; border-radius: 10px;")
        self.lbl_mc_desc = QLabel("설명 텍스트")
        self.lbl_mc_desc.setWordWrap(True)
        self.lbl_mc_desc.setAlignment(Qt.AlignCenter)
        self.lbl_mc_desc.setStyleSheet("font-size: 32px; font-weight: bold; color: white; background-color: #2b2b2b; padding: 30px; border-radius: 10px;")
        self.mc_content_stack.addWidget(self.lbl_mc_image)
        self.mc_content_stack.addWidget(self.lbl_mc_desc)
        grid_layout = QGridLayout()
        self.btn_mc_choices = []
        for i in range(4):
            btn = QPushButton(f"보기 {i+1}")
            btn.setStyleSheet("background-color: #008CBA; color: white; font-size: 24px; font-weight: bold; padding: 20px 30px; text-align: left; padding-left: 30px; border-radius: 10px;")
            btn.clicked.connect(lambda checked, idx=i: self.check_mc_answer(idx))
            self.btn_mc_choices.append(btn)
            grid_layout.addWidget(btn, i//2, i%2)
        self.btn_exit_quiz2 = QPushButton("퀴즈 메뉴로")
        self.btn_exit_quiz2.setStyleSheet("background-color: #f44336; color: white; font-size: 20px; font-weight: bold; padding: 15px; border-radius: 10px;")
        self.btn_exit_quiz2.clicked.connect(self.exit_mc_quiz)
        mc_layout.addWidget(self.lbl_quiz_title_mc)
        mc_layout.addWidget(self.speed_progress)
        mc_layout.addWidget(self.mc_content_stack, stretch=1)
        mc_layout.addLayout(grid_layout)
        mc_layout.addWidget(self.btn_exit_quiz2)
        self.test_stack.addWidget(menu_widget)
        self.test_stack.addWidget(action_widget)
        self.test_stack.addWidget(mc_widget)
        self.tab_test.setLayout(QVBoxLayout())
        self.tab_test.layout().addWidget(self.test_stack)

    def setup_auto_tab(self):
        layout = QVBoxLayout()
        cam_panel = QVBoxLayout()
        lbl_auto_cam = QLabel("자율 동작 녹화 및 AI 번역 모드")
        lbl_auto_cam.setStyleSheet("font-size: 20px; font-weight: bold; color: #008CBA;")
        self.lbl_auto_camera = QLabel("카메라 구동 대기 중...")
        self.lbl_auto_camera.setAlignment(Qt.AlignCenter)
        self.lbl_auto_camera.setStyleSheet("background-color: black; border-radius: 10px;")
        cam_panel.addWidget(lbl_auto_cam)
        cam_panel.addWidget(self.lbl_auto_camera, stretch=1)
        
        bottom_panel = QVBoxLayout()
        lbl_local_title = QLabel("AI 판별 및 피드백 (사전 외의 동작도 뼈대 데이터와 시각 정보를 통해 유추하여 번역합니다)")
        lbl_local_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")
        self.lbl_auto_result = QLabel("준비가 되면 아래 버튼을 누르고 2초간 수어 동작을 취해보세요.")
        self.lbl_auto_result.setWordWrap(True)
        self.lbl_auto_result.setAlignment(Qt.AlignCenter)
        self.lbl_auto_result.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background-color: #2b2b2b; border-radius: 10px; padding: 20px;")
        
        btn_layout_auto = QHBoxLayout()
        self.btn_auto_scan = QPushButton("동작 2초 녹화 후 AI 번역")
        self.btn_auto_scan.setStyleSheet("background-color: #4CAF50; color: white; font-size: 22px; font-weight: bold; padding: 15px; border-radius: 8px;")
        self.btn_auto_scan.clicked.connect(lambda: self.start_recording("auto"))
        
        self.btn_stop_audio_auto = QPushButton("⏹️ 음성 종료")
        self.btn_stop_audio_auto.setStyleSheet("background-color: #f44336; color: white; font-size: 22px; font-weight: bold; padding: 15px; border-radius: 8px;")
        self.btn_stop_audio_auto.clicked.connect(self.stop_tts_audio)
        
        btn_layout_auto.addWidget(self.btn_auto_scan, stretch=3)
        btn_layout_auto.addWidget(self.btn_stop_audio_auto, stretch=1)
        
        bottom_panel.addWidget(lbl_local_title)
        bottom_panel.addWidget(self.lbl_auto_result, stretch=1)
        bottom_panel.addLayout(btn_layout_auto)
        layout.addLayout(cam_panel, 7)
        layout.addLayout(bottom_panel, 3)
        self.tab_auto.setLayout(layout)

    def get_image_path(self, word):
        if not os.path.exists(self.base_path): return None
        clean_target_word = unicodedata.normalize('NFC', word.strip().replace(" ", ""))
        for file in os.listdir(self.base_path):
            name, ext = os.path.splitext(file)
            clean_name = unicodedata.normalize('NFC', name.strip().replace(" ", ""))
            ext = ext.lower()
            if clean_target_word == clean_name and ext in ['.jpg', '.jpeg', '.png']:
                return os.path.join(self.base_path, file)
        return None

    def play_video(self):
        if hasattr(self, 'current_video_path') and self.current_video_path:
            safe_video_path = "/tmp/temp_sign_video.mp4"
            shutil.copy2(self.current_video_path, safe_video_path)
            self.learn_media_player.setMedia(QMediaContent(QUrl.fromLocalFile(safe_video_path)))
            self.learn_video_stack.setCurrentIndex(1)
            self.learn_media_player.play()

    def replay_video(self):
        print(f"🎬 [UI 액션] 영상 다시보기 실행: {self.current_target_word}", flush=True)
        self.play_video()

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.learn_video_stack.setCurrentIndex(0)

    def start_quiz_session(self, q_type):
        print(f"🎮 [시스템] 퀴즈 모드 시작: {q_type}", flush=True)
        self.quiz_type = q_type
        self.quiz_current_q = 1
        self.quiz_correct_count = 0
        self.speed_timer.stop()
        self.stop_tts_audio()
        self.speed_progress.hide()
        if q_type in ["action", "initial", "survival"]:
            self.test_stack.setCurrentIndex(1)
            self.next_action_question()
        else:
            self.test_stack.setCurrentIndex(2)
            self.next_mc_question()

    def next_action_question(self):
        if self.quiz_type != "survival" and self.quiz_current_q > 5:
            self.show_quiz_result()
            return
        self.quiz_target_word = random.choice(self.words)
        if self.quiz_type == "action":
            title = f"실전 동작 퀴즈 (문제 {self.quiz_current_q}/5)"
            target_text = f"제시어: [{self.quiz_target_word}]"
        elif self.quiz_type == "initial":
            title = f"🔠 초성 동작 퀴즈 (문제 {self.quiz_current_q}/5)"
            target_text = f"초성: [{get_chosung(self.quiz_target_word)}]"
        elif self.quiz_type == "survival":
            title = f"💀 서바이벌 모드 (현재 {self.quiz_correct_count}연속 성공 중)"
            target_text = f"제시어: [{self.quiz_target_word}]"
        self.lbl_quiz_title_act.setText(title)
        self.lbl_quiz_target_act.setText(target_text)
        self.lbl_quiz_target_act.setStyleSheet("font-size: 36px; font-weight: bold; color: #FFD700; background-color: #2b2b2b; padding: 20px; border-radius: 15px;")
        self.btn_submit_act.setEnabled(True)

    def update_speed_timer(self):
        self.speed_time_left -= 1
        self.speed_progress.setValue(self.speed_time_left)
        if self.speed_time_left > 0 and self.speed_time_left % 10 == 0:
            self.play_sound_effect("타이머")
        if self.speed_time_left <= 0:
            self.speed_timer.stop()
            self.play_sound_effect("오답")
            self.handle_mc_wrong_answer()

    def handle_mc_wrong_answer(self):
        for btn in self.btn_mc_choices: btn.setEnabled(False)
        for i, w in enumerate(self.mc_current_choices):
            if w == self.quiz_target_word:
                self.btn_mc_choices[i].setStyleSheet("background-color: #4CAF50; color: white; font-size: 18px; font-weight: bold; padding: 20px; border-radius: 10px; text-align: left; padding-left: 30px;")
            else:
                self.btn_mc_choices[i].setStyleSheet("background-color: #f44336; color: white; font-size: 18px; font-weight: bold; padding: 20px; border-radius: 10px; text-align: left; padding-left: 30px;")
        self.quiz_current_q += 1
        QTimer.singleShot(1500, self.next_mc_question)

    def exit_mc_quiz(self):
        print("🚪 [UI 액션] 퀴즈 종료 및 나가기", flush=True)
        self.speed_timer.stop()
        self.stop_tts_audio()
        self.test_stack.setCurrentIndex(0)

    def next_mc_question(self):
        if self.quiz_current_q > 5:
            self.show_quiz_result()
            return
        self.quiz_target_word = random.choice(self.words)
        choices = random.sample([w for w in self.words if w != self.quiz_target_word], 3)
        choices.append(self.quiz_target_word)
        random.shuffle(choices)
        self.mc_current_choices = choices
        if self.quiz_type == "reverse":
            self.lbl_quiz_title_mc.setText(f"거꾸로 설명 맞추기 (문제 {self.quiz_current_q}/5)")
            for i in range(4):
                desc = self.word_data.get(choices[i], "설명 없음")
                if len(desc) > 30: desc = desc[:30] + "..."
                self.btn_mc_choices[i].setText(desc)
                self.btn_mc_choices[i].setStyleSheet("background-color: #008CBA; color: white; font-size: 16px; font-weight: bold; padding: 15px; border-radius: 10px; text-align: left; padding-left: 30px;")
                self.btn_mc_choices[i].setEnabled(True)
            img_path = self.get_image_path(self.quiz_target_word)
            if img_path:
                pixmap = QPixmap(img_path).scaled(450, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.lbl_mc_image.setPixmap(pixmap)
            else:
                self.lbl_mc_image.setText(f"[{self.quiz_target_word}] 사진 없음")
            self.mc_content_stack.setCurrentIndex(0)
        else:
            for i in range(4):
                self.btn_mc_choices[i].setText(choices[i])
                self.btn_mc_choices[i].setStyleSheet("background-color: #008CBA; color: white; font-size: 24px; font-weight: bold; padding: 20px; border-radius: 10px; text-align: left; padding-left: 30px;")
                self.btn_mc_choices[i].setEnabled(True)
            if self.quiz_type in ["image", "speed"]:
                title = f"스피드 단어 맞추기 (문제 {self.quiz_current_q}/5)" if self.quiz_type == "speed" else f"이미지 매칭 퀴즈 (문제 {self.quiz_current_q}/5)"
                self.lbl_quiz_title_mc.setText(title)
                img_path = self.get_image_path(self.quiz_target_word)
                if img_path:
                    pixmap = QPixmap(img_path).scaled(450, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.lbl_mc_image.setPixmap(pixmap)
                else:
                    self.lbl_mc_image.setText(f"[{self.quiz_target_word}] 사진 없음")
                self.mc_content_stack.setCurrentIndex(0)
                if self.quiz_type == "speed":
                    self.speed_progress.show()
                    self.speed_time_left = 50
                    self.speed_progress.setValue(50)
                    self.speed_timer.start(100)
            elif self.quiz_type == "desc":
                self.lbl_quiz_title_mc.setText(f"설명 매칭 퀴즈 (문제 {self.quiz_current_q}/5)")
                desc = self.word_data.get(self.quiz_target_word, "설명이 없습니다.")
                self.lbl_mc_desc.setText(f"설명: {desc}")
                self.mc_content_stack.setCurrentIndex(1)

    def check_mc_answer(self, btn_idx):
        self.speed_timer.stop()
        self.stop_tts_audio()
        selected_word = self.mc_current_choices[btn_idx]
        for btn in self.btn_mc_choices: btn.setEnabled(False)
        font_size = "16px" if self.quiz_type == "reverse" else "24px"
        if selected_word == self.quiz_target_word:
            self.play_sound_effect("정답")
            self.quiz_correct_count += 1
            self.btn_mc_choices[btn_idx].setStyleSheet(f"background-color: #4CAF50; color: white; font-size: {font_size}; font-weight: bold; padding: 20px; border-radius: 10px; text-align: left; padding-left: 30px;")
        else:
            self.play_sound_effect("오답")
            self.btn_mc_choices[btn_idx].setStyleSheet(f"background-color: #f44336; color: white; font-size: {font_size}; font-weight: bold; padding: 20px; border-radius: 10px; text-align: left; padding-left: 30px;")
            for i, w in enumerate(self.mc_current_choices):
                if w == self.quiz_target_word:
                    self.btn_mc_choices[i].setStyleSheet(f"background-color: #4CAF50; color: white; font-size: {font_size}; font-weight: bold; padding: 20px; border-radius: 10px; text-align: left; padding-left: 30px;")
        self.quiz_current_q += 1
        QTimer.singleShot(1500, self.next_mc_question)

    def show_quiz_result(self):
        print(f"📊 [시스템] 퀴즈 종료 (맞춘 개수: {self.quiz_correct_count})", flush=True)
        self.play_sound_effect("종료")
        total = float('inf') if self.quiz_type == "survival" else 5
        dialog = QuizResultDialog(self.quiz_correct_count, total, self)
        dialog.exec_()
        self.test_stack.setCurrentIndex(0)

    def start_recording(self, mode):
        print(f"🎥 [UI 액션] 녹화 버튼 클릭됨 - 모드: {mode}", flush=True)
        self.stop_tts_audio()
        self.analysis_mode = mode
        self.is_counting_down = True
        self.countdown_start_time = time.time()
        
        self.last_pose = [0.0]*12
        self.last_lh = [0.0]*15
        self.last_rh = [0.0]*15
        self.last_rel = [0.0]*9
        self.last_rel_head = [0.0]*6
        self.last_torso = [0.0]*18
        
        if mode == "practice":
            self.record_duration = 2.0
            self.btn_check_motion.setEnabled(False)
            self.practice_progress.setValue(0)
            self.practice_progress.setFormat("⏳ 원 안으로 두 손을 모으고 준비하세요... 3, 2, 1")
            self.lbl_practice_ai_feedback.setText("...")
        elif mode in ["quiz_act", "initial", "survival"]:
            self.record_duration = 1.0 
            self.btn_submit_act.setEnabled(False)
            self.lbl_quiz_target_act.setText("⏳ 동작 준비... (3초 후 시작)")
            self.lbl_quiz_target_act.setStyleSheet("background-color: #FF9800; color: white; font-size: 36px; font-weight: bold; padding: 20px; border-radius: 15px;")
        elif mode == "auto":
            self.record_duration = 2.0
            self.btn_auto_scan.setEnabled(False)
            self.btn_auto_scan.setText("⏳ 준비하세요... 3, 2, 1")
            self.btn_auto_scan.setStyleSheet("background-color: #FF9800; color: white; font-size: 22px; font-weight: bold; padding: 15px; border-radius: 8px;")

    def get_3d_direction(self, p1, p2):
        v = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
        norm = np.linalg.norm(v)
        return (v / norm).tolist() if norm > 0 else [0.0, 0.0, 0.0]

    def extract_feature_vector(self, results):
        vec = []
        lms_pose = results.pose_landmarks.landmark if results.pose_landmarks else None
        lms_lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
        lms_rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None

        if lms_pose:
            pose = []
            pose.extend(self.get_3d_direction(lms_pose[11], lms_pose[13])) 
            pose.extend(self.get_3d_direction(lms_pose[12], lms_pose[14])) 
            pose.extend(self.get_3d_direction(lms_pose[13], lms_pose[15])) 
            pose.extend(self.get_3d_direction(lms_pose[14], lms_pose[16])) 
            self.last_pose = pose
        vec.extend(self.last_pose)

        if lms_lh:
            lh = []
            for tip in [4, 8, 12, 16, 20]: lh.extend(self.get_3d_direction(lms_lh[0], lms_lh[tip]))
            self.last_lh = lh
        vec.extend(self.last_lh)

        if lms_rh:
            rh = []
            for tip in [4, 8, 12, 16, 20]: rh.extend(self.get_3d_direction(lms_rh[0], lms_rh[tip]))
            self.last_rh = rh
        vec.extend(self.last_rh)

        if lms_pose and len(lms_pose) > 24: 
            cx = (lms_pose[11].x + lms_pose[12].x) / 2
            cy = (lms_pose[11].y + lms_pose[12].y) / 2
            cz = (lms_pose[11].z + lms_pose[12].z) / 2
            class Point3D:
                def __init__(self, x, y, z):
                    self.x = x; self.y = y; self.z = z
            center = Point3D(cx, cy, cz)
            
            rel = []
            rel.extend(self.get_3d_direction(lms_pose[15], lms_pose[16])) 
            rel.extend(self.get_3d_direction(center, lms_pose[15]))       
            rel.extend(self.get_3d_direction(center, lms_pose[16]))       
            self.last_rel = rel
            
            head = lms_pose[0]
            rel_head = []
            rel_head.extend(self.get_3d_direction(head, lms_pose[15])) 
            rel_head.extend(self.get_3d_direction(head, lms_pose[16])) 
            self.last_rel_head = rel_head

            sx = (lms_pose[11].x + lms_pose[12].x + lms_pose[23].x + lms_pose[24].x) / 4
            sy = (lms_pose[11].y + lms_pose[12].y + lms_pose[23].y + lms_pose[24].y) / 4
            sz = (lms_pose[11].z + lms_pose[12].z + lms_pose[23].z + lms_pose[24].z) / 4
            stomach = Point3D(sx, sy, sz)

            torso = []
            torso.extend(self.get_3d_direction(lms_pose[11], lms_pose[15])) 
            torso.extend(self.get_3d_direction(lms_pose[11], lms_pose[16])) 
            torso.extend(self.get_3d_direction(lms_pose[12], lms_pose[15])) 
            torso.extend(self.get_3d_direction(lms_pose[12], lms_pose[16])) 
            torso.extend(self.get_3d_direction(stomach, lms_pose[15]))      
            torso.extend(self.get_3d_direction(stomach, lms_pose[16]))      
            self.last_torso = torso
        
        vec.extend(self.last_rel)
        vec.extend(self.last_rel_head)
        vec.extend(self.last_torso) 

        return vec

    def resample_sequence(self, seq, target_len=30):
        if seq is None or len(seq) < 2: return None
        seq_arr = np.array(seq, dtype=np.float32)
        if len(seq_arr.shape) != 2 or seq_arr.shape[1] != 75: 
            print(f"⚠️ [경고] 데이터 규격 오류: {seq_arr.shape[1]}차원 발견 (75차원 강제)", flush=True)
            return None 
        n_frames, n_dims = seq_arr.shape
        if n_frames == target_len: return seq_arr
        orig_idx = np.linspace(0, 1, n_frames)
        target_idx = np.linspace(0, 1, target_len)
        resampled = np.zeros((target_len, n_dims), dtype=np.float32)
        for d in range(n_dims):
            resampled[:, d] = np.interp(target_idx, orig_idx, seq_arr[:, d])
        return resampled

    def get_dynamic_weights(self, word):
        face_words = ["미안합니다", "좋다", "기다리다", "나이", "정말로", "핸드폰", "못생기다", "괜찮다", "남자", "여자"]
        torso_words = ["일등", "필요", "나", "감사합니다"]
        rel_words = ["안녕하세요", "만나다", "돕다", "책", "아니다"]
        
        if word in face_words:
            w = np.array([0.5]*12 + [2.0]*30 + [1.0]*9 + [5.0]*6 + [0.5]*18)
        elif word in torso_words:
            w = np.array([0.5]*12 + [2.0]*30 + [1.0]*9 + [1.0]*6 + [5.0]*18)
        elif word in rel_words:
            w = np.array([0.5]*12 + [2.0]*30 + [6.0]*9 + [1.0]*6 + [0.5]*18)
        else: 
            w = np.array([0.2]*12 + [5.0]*30 + [0.5]*9 + [0.5]*6 + [0.2]*18)
            
        return (w / np.sum(w)) * 75.0

    def compute_universal_dtw(self, seq1, seq2, weights):
        n, m = len(seq1), len(seq2)
        if n == 0 or m == 0: return float('inf')
        seq1_arr = np.array(seq1)
        seq2_arr = np.array(seq2)
        
        diff = seq1_arr[:, None, :] - seq2_arr[None, :, :]
        sq_diff = diff ** 2
        
        sq_diff = np.clip(sq_diff, 0, 2.0)
        
        weighted_sq_diff = sq_diff * weights
        dist_matrix = np.sqrt(np.sum(weighted_sq_diff, axis=2))
        
        w = 4
        dtw_matrix = np.full((n + 1, m + 1), float('inf'))
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            j_start = max(1, i - w)
            j_end = min(m + 1, i + w + 1)
            for j in range(j_start, j_end):
                cost = dist_matrix[i-1, j-1]
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
                
        return dtw_matrix[n, m] / n

    def evaluate_motion_locally(self, user_seq):
        print(f"🧠 [라즈베리파이 정밀 연산] 75차원 가중치 타임 정규화 DTW 대조 시작", flush=True)
        try:
            if len(user_seq) < 5: return [("프레임 부족", 0), ("-", 0), ("-", 0)]
            user_arr = np.array(user_seq)
            
            motion_energy = np.mean(np.std(user_arr, axis=0))
            if motion_energy < 0.001:
                return [("동작 대기 중...", 0), ("-", 0), ("-", 0)]
            
            norm_user = self.resample_sequence(user_seq, 30)
            if norm_user is None: return [("데이터 규격 불일치 (정답지 갱신 필요)", 0), ("-", 0), ("-", 0)]

            results = []
            for word in self.words:
                if word not in self.expert_data: continue
                expert_seq = self.expert_data[word] 
                if not expert_seq or len(expert_seq) == 0: continue
                
                dynamic_weights = self.get_dynamic_weights(word)
                dist = self.compute_universal_dtw(norm_user, expert_seq, dynamic_weights)
                
                if dist == float('inf'): 
                    score = 0
                else:
                    base_dist = max(0, dist - 1.5) 
                    converted_score = 100 - (base_dist * 10.0) 
                    score = int(max(0, min(100, converted_score)))
                results.append((word, score))
                
            results.sort(key=lambda x: x[1], reverse=True)
            while len(results) < 3: results.append(("알 수 없음", 0))
            
            if results[0][1] > 35 and (results[0][1] - results[1][1] > 10):
                adjusted_score = min(100, results[0][1] + 30)
                results[0] = (results[0][0], adjusted_score)

            print(f"🏆 [로컬 연산 완료] 1위 예측: '{results[0][0]}' ({results[0][1]}점)", flush=True)
            return results[:3]
        except Exception as e:
            traceback.print_exc()
            print(f"❌ [로컬 연산 오류] {e}", flush=True)
            return [("연산 오류", 0), ("-", 0), ("-", 0)]

    def process_hybrid_analysis(self, user_seq=None, vision_frame=None):
        if user_seq is None: user_seq = list(self.pose_buffer)
        top3 = self.evaluate_motion_locally(user_seq)
        self.ui_queue.put(("local", top3, self.analysis_mode))
        best_match, best_score = top3[0]
        
        if self.analysis_mode in ["action", "initial", "survival", "quiz_act"]:
            feedback = "로컬 채점 완료"
            self.ui_queue.put(("ai", feedback, top3, self.analysis_mode))
        elif self.analysis_mode == "auto":
            if best_match in ["손 동작 없음", "동작 대기 중...", "데이터 규격 불일치 (정답지 갱신 필요)", "연산 오류"]:
                feedback = "수어 동작이 감지되지 않았습니다. (AI 전송 차단)"
                self.ui_queue.put(("ai", feedback, top3, self.analysis_mode))
            elif best_score >= 50: 
                feedback, _, _ = self.request_gemini_feedback(top3, "auto_known", None, user_seq)
                self.ui_queue.put(("ai", feedback, top3, self.analysis_mode))
            else:
                feedback, _, _ = self.request_gemini_feedback(top3, "auto_unknown", vision_frame, user_seq)
                self.ui_queue.put(("ai", feedback, top3, self.analysis_mode))
        else: 
            if best_match in ["손 동작 없음", "동작 대기 중...", "데이터 규격 불일치 (정답지 갱신 필요)", "연산 오류"]:
                feedback = "수어 동작이 감지되지 않았습니다. 다시 동작해주세요."
                self.ui_queue.put(("ai", feedback, top3, self.analysis_mode))
            else:
                feedback, _, _ = self.request_gemini_feedback(top3, self.analysis_mode, vision_frame, user_seq)
                self.ui_queue.put(("ai", feedback, top3, self.analysis_mode))

    def update_ui_local_first(self, top3, mode):
        best_match, best_score = top3[0]
        if mode == "practice":
            target = self.current_target_word
            target_score = next((score for w, score in top3 if w == target), 0)
            self.practice_progress.setValue(target_score)
            self.practice_progress.setFormat(f"로컬 판독: {target} ({target_score}점) - 구글 AI 전송 중...")
        elif mode == "auto":
            if best_match in ["손 동작 없음", "동작 대기 중...", "데이터 규격 불일치 (정답지 갱신 필요)", "연산 오류"]:
                self.lbl_auto_result.setText("⚠️ 동작이 감지되지 않아 분석을 취소했습니다.")
            elif best_score >= 50:
                self.lbl_auto_result.setText(f"✅ 로컬 사전 매칭 완료: {best_match}\n⏳ 구글 AI가 문장을 다듬고 있습니다...")
            else:
                self.lbl_auto_result.setText(f"✅ 녹화 완료! (가장 유사한 동작: {best_match})\n⏳ 구글 AI 비전이 뼈대 데이터와 사진을 종합 분석/번역 중입니다...")

    def request_gemini_feedback(self, top3, mode, vision_frame=None, user_seq=None):
        print(f"📡 [구글 AI 연동] 프롬프트 엔지니어링 및 네트워크 전송 시작", flush=True)
        best_match, best_score = top3[0]
        prompt = ""
        motion_info = ""
        if user_seq is not None and len(user_seq) > 0:
            user_arr = np.array(user_seq)
            try:
                rh_energy = np.mean(np.std(user_arr[:, 27:42], axis=0))
                lh_energy = np.mean(np.std(user_arr[:, 12:27], axis=0))
                body_energy = np.mean(np.std(user_arr[:, 0:12], axis=0))
                norm_seq = self.resample_sequence(user_seq, 3)
                if norm_seq is not None:
                    start_vec = np.round(norm_seq[0], 2).tolist()
                    mid_vec = np.round(norm_seq[1], 2).tolist()
                    end_vec = np.round(norm_seq[2], 2).tolist()
                    vector_str = f"시작 프레임: {start_vec[:42]}\n중간 프레임: {mid_vec[:42]}\n종료 프레임: {end_vec[:42]}"
                else: vector_str = "데이터 추출 불가"
                motion_info = f"""
[MediaPipe 3D 뼈대 추적(Landmark) 및 움직임 수치 정보]
- 오른손 활성도 (수치가 클수록 많이 움직임): {rh_energy:.4f}
- 왼손 활성도: {lh_energy:.4f}
- 상체/어깨 활성도: {body_energy:.4f}
- 핵심 3프레임(시작/중간/종료)의 3D 벡터 변화량:
{vector_str}
"""
            except Exception: motion_info = "[뼈대 수치 정보 추출 실패]"

        if mode == "practice":
            target = self.current_target_word
            target_score = next((score for w, score in top3 if w == target), 0)
            desc = self.word_data.get(target, "")
            prompt = f"수어 강사로서 친절하게 피드백해줘. 학생이 '{target}'를 연습했고 로컬 점수는 {target_score}점이야. 올바른 방법('{desc}')을 참고하고, 첨부된 사진과 아래의 뼈대 수치 데이터를 분석해서 교정할 점을 구체적이고 상세하게 짚어줘.\n{motion_info}"
        elif mode == "auto_known":
            prompt = f"""사용자가 수어 '{best_match}' 동작을 한 것으로 시스템이 확신합니다(정확도 {best_score}점). 
아래의 움직임 활성도를 참고하여, 이 단어를 자연스러운 한국어 문장으로 다듬어주세요.
{motion_info}
출력 형식은 반드시 아래와 같이 두 줄로 작성해 주세요.
번역 결과: [내용]
판단 근거: [제공된 뼈대 수치와 '{best_match}'의 실제 수형(손모양/위치) 특징을 연결지어, 왜 이 동작이 완벽하게 일치하는지 구체적이고 상세하게 설명]"""
            mode = "auto"
        elif mode == "auto_unknown":
            prompt = f"""첨부된 사진은 사용자가 수어를 하는 시작, 중간, 끝 3프레임 연속 사진입니다.
구글 AI의 완벽한 판단을 돕기 위해 라즈베리파이에서 추출한 '모든' 시스템 및 뼈대 데이터를 함께 제공합니다.

[시스템 1차 로컬 엔진 예측 (참고용)]
- 1위 예측: {top3[0][0]} ({top3[0][1]}점)
- 2위 예측: {top3[1][0]} ({top3[1][1]}점)
- 3위 예측: {top3[2][0]} ({top3[2][1]}점)
{motion_info}

로컬 매칭 점수가 낮아(50점 미만) 사전에 없는 동작이거나 비수어 동작일 확률이 높습니다.
첨부된 시각 자료(사진)와 위의 수치화된 뼈대 데이터를 모두 종합하여 이 동작의 진짜 의미를 정밀하게 유추해 주세요. 
단, 분석 결과 수어가 아니거나 단순한 일상 행동이라면 억지로 유추하지 말고 '수어 동작이 아님'이라고 명확히 답변하세요.

출력 형식은 반드시 아래의 두 줄 포맷으로 작성해 주세요.
번역 결과: [자연스러운 한국어 번역 내용 또는 '수어 동작이 아님']
판단 근거: [제공된 뼈대 활성도 수치, 손의 위치, 이동 궤적을 구체적으로 언급하며 이 결론을 내린 이유를 상세하게 설명]"""
            mode = "auto"

        print("\n" + "┏" + "━"*72 + "┓", flush=True)
        print("┃ 📡 [Gemini AI 통신 브리핑 및 프롬프트 투명 공개]                 ┃", flush=True)
        print("┣" + "━"*72 + "┫", flush=True)
        print(f"  ▶ 작동 모드: {mode}", flush=True)
        print(f"  ▶ 로컬 판단 1위: '{best_match}' (정확도 {best_score}점)", flush=True)
        if vision_frame is not None:
            print(f"  ▶ 첨부 데이터: 3프레임 컷 이미지 전송 O (해상도 {vision_frame.shape[1]}x{vision_frame.shape[0]})", flush=True)
        else:
            print(f"  ▶ 첨부 데이터: 이미지 전송 X (고속 텍스트망 이용)", flush=True)
        print(f"  ▶ [전송되는 명령어(Prompt) 전문]\n{prompt}", flush=True)
        print("┗" + "━"*72 + "┛\n", flush=True)

        try:
            if not GEMINI_READY: raise ValueError("API 라이브러리 초기화 실패")
            start_time = time.time()
            if vision_frame is not None and PILLOW_READY and mode in ["practice", "auto"]:
                img_pil = PILImage.fromarray(vision_frame)
                response = gemini_model.generate_content([prompt, img_pil])
            else:
                response = gemini_model.generate_content(prompt)
            elapsed = time.time() - start_time
            feedback = response.text.strip()
            print(f"✅ [Gemini AI 응답 완료] ({elapsed:.2f}초 소요)", flush=True)
            print(f"  ▶ 수신된 원본 텍스트:\n{feedback}", flush=True)
            print("="*74 + "\n", flush=True)
        except Exception as e:
            print(f"❌ [Gemini AI 통신 오류 발생] {e}", flush=True)
            feedback = f"통신 지연 (자체 로컬 엔진 판독 완료: {best_match})"
        return feedback, best_match, best_score

    def update_ui_ai_feedback(self, feedback, top3, mode):
        best_match, best_score = top3[0]
        if mode == "practice":
            self.btn_check_motion.setEnabled(True)
            target = self.current_target_word
            target_score = next((score for w, score in top3 if w == target), 0)
            self.practice_progress.setFormat(f"최종 정확도: {target_score}점")
            self.practice_progress.setValue(target_score)
            self.lbl_practice_ai_feedback.setText(f"🤖 구글 AI 피드백:\n{feedback}")
            self.speak_text(feedback)
        elif mode in ["action", "initial", "survival", "quiz_act"]: 
            if best_match == self.quiz_target_word and best_score >= 60: 
                self.play_sound_effect("정답")
                self.quiz_correct_count += 1
                self.lbl_quiz_target_act.setText(f"✅ 정답! (빠른 로컬 판독: {best_score}점)")
                self.lbl_quiz_target_act.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background-color: #4CAF50; padding: 20px; border-radius: 15px;")
            else:
                self.play_sound_effect("오답")
                self.lbl_quiz_target_act.setText(f"❌ 오답 (내 동작: {best_match} / {best_score}점)")
                self.lbl_quiz_target_act.setStyleSheet("font-size: 24px; font-weight: bold; color: white; background-color: #FF5722; padding: 20px; border-radius: 15px;")
            if mode == "survival" and best_match != self.quiz_target_word:
                QTimer.singleShot(2500, self.show_quiz_result)
            else:
                self.quiz_current_q += 1
                QTimer.singleShot(2500, self.next_action_question)
        elif mode == "auto":
            self.btn_auto_scan.setEnabled(True)
            self.btn_auto_scan.setText("동작 2초 녹화 후 AI 번역")
            self.btn_auto_scan.setStyleSheet("background-color: #4CAF50; color: white; font-size: 22px; font-weight: bold; padding: 15px; border-radius: 8px;")
            if "차단" in feedback:
                self.lbl_auto_result.setText(f"🛑 판독 취소: {feedback}")
            else:
                self.lbl_auto_result.setText(f"🤖 구글 AI 분석 리포트:\n\n{feedback}")
            if "번역 결과:" in feedback:
                speech_text = feedback.split("번역 결과:")[1].split("판단 근거:")[0].strip()
                self.speak_text(speech_text)

    def update_camera_frame(self):
        if self.root_stack.currentIndex() == 0: return

        while not self.ui_queue.empty():
            try:
                msg = self.ui_queue.get_nowait()
                if msg[0] == "local": self.update_ui_local_first(msg[1], msg[2])
                elif msg[0] == "ai": self.update_ui_ai_feedback(msg[1], msg[2], msg[3])
            except Exception as e:
                print(f"❌ [UI 큐 처리 에러] {e}", flush=True)

        if getattr(self, 'is_processing_frame', False): return
        self.is_processing_frame = True
        
        try:
            if self.cap is None or not self.cap.isOpened(): return
            self.frame_count += 1
            ret, frame = self.cap.read()
            if not ret: return
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w

            need_holistic = True
            current_tab = self.tabs.currentIndex()
            if current_tab == 0: need_holistic = False
                
            results = None
            if need_holistic:
                results = self.holistic.process(rgb_frame)
            
            if results:
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    self.mp_drawing.draw_landmarks(rgb_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    self.mp_drawing.draw_landmarks(rgb_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    
                current_vector = self.extract_feature_vector(results)
                
                if self.is_recording:
                    self.pose_buffer.append(current_vector)
                    if not hasattr(self, 'vision_buffer'): self.vision_buffer = []
                    self.vision_buffer.append(rgb_frame.copy())
                    if time.time() - self.record_start_time >= getattr(self, 'record_duration', 2.0):
                        print("⏹️ [녹화 시스템] 사용자 동작 캡처 종료. 데이터 하이브리드 엔진 전송 중...", flush=True)
                        self.is_recording = False
                        seq_copy = list(self.pose_buffer)
                        if hasattr(self, 'vision_buffer') and len(self.vision_buffer) >= 3:
                            f1 = self.vision_buffer[0]
                            f2 = self.vision_buffer[len(self.vision_buffer)//2]
                            f3 = self.vision_buffer[-1]
                            vision_copy = cv2.hconcat([f1, f2, f3])
                        else: vision_copy = rgb_frame.copy()
                        self.vision_buffer = [] 
                        threading.Thread(target=self.process_hybrid_analysis, args=(seq_copy, vision_copy), daemon=True).start()

            final_frame = cv2.flip(rgb_frame, 1)

            if getattr(self, 'is_counting_down', False):
                elapsed = time.time() - self.countdown_start_time
                remain = 3 - int(elapsed)
                if remain > 0:
                    center_x, center_y = int(w/2), int(h/2)
                    cv2.putText(final_frame, str(remain), (center_x - 30, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 165, 255), 10, cv2.LINE_AA)
                else:
                    self.is_counting_down = False
                    self.is_recording = True
                    self.record_start_time = time.time()
                    self.pose_buffer.clear()
                    self.vision_buffer = []
                    print("🔴 [녹화 시스템] 사용자 동작 2초 캡처 진입...", flush=True)
                    self.play_sound_effect("안내멘트")
                    if self.analysis_mode == "practice":
                        self.ui_queue.put(("ui_update", "practice"))
                        self.practice_progress.setFormat("🔴 동작을 녹화 중입니다...")
                    elif self.analysis_mode in ["quiz_act", "initial", "survival"]:
                        self.lbl_quiz_target_act.setText("🔴 1초 빠른 판독 진행 중...")
                        self.lbl_quiz_target_act.setStyleSheet("background-color: #f44336; color: white; font-size: 36px; font-weight: bold; padding: 20px; border-radius: 15px;")
                    elif self.analysis_mode == "auto":
                        self.lbl_auto_result.setText("카메라를 보고 자유롭게 수어 동작을 취해주세요...")
                        self.btn_auto_scan.setText("🔴 동작을 녹화 중입니다...")
                        self.btn_auto_scan.setStyleSheet("background-color: #f44336; color: white; font-size: 22px; font-weight: bold; padding: 15px; border-radius: 8px;")

            if getattr(self, 'is_counting_down', False): box_color = (0, 165, 255) 
            elif self.is_recording: box_color = (255, 0, 0)
            else: box_color = (0, 255, 0)
                
            cv2.rectangle(final_frame, (int(w*0.02), int(h*0.02)), (int(w*0.98), int(h*0.98)), box_color, 4 if (self.is_recording or getattr(self, 'is_counting_down', False)) else 2)
            
            center_x = int(w / 2)
            head_y = int(h * 0.25)
            head_r_x, head_r_y = int(w * 0.08), int(h * 0.15)
            shoulder_y = int(h * 0.60)
            shoulder_w = int(w * 0.28)
            
            cv2.ellipse(final_frame, (center_x, head_y), (head_r_x, head_r_y), 0, 0, 360, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(final_frame, "HEAD", (center_x - 30, head_y - head_r_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.line(final_frame, (center_x - shoulder_w, shoulder_y), (center_x + shoulder_w, shoulder_y), (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(final_frame, "SHOULDER", (center_x - 55, shoulder_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            
            hand_y = int(h * 0.88)
            hand_offset = int(w * 0.21)
            hand_r = int(w * 0.045)
            cv2.circle(final_frame, (center_x - hand_offset, hand_y), hand_r, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(final_frame, "L-HAND", (center_x - hand_offset - 30, hand_y - hand_r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(final_frame, (center_x + hand_offset, hand_y), hand_r, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(final_frame, "R-HAND", (center_x + hand_offset - 30, hand_y - hand_r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA)
            
            q_img = QImage(final_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            scaled_img = QPixmap.fromImage(q_img)
            
            current_tab = self.tabs.currentIndex()
            if current_tab == 1 and hasattr(self, 'lbl_practice_camera'):
                self.lbl_practice_camera.setPixmap(scaled_img.scaled(self.lbl_practice_camera.width(), self.lbl_practice_camera.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            elif current_tab == 2 and hasattr(self, 'lbl_test_camera') and self.quiz_type in ["action", "initial", "survival"]:
                self.lbl_test_camera.setPixmap(scaled_img.scaled(self.lbl_test_camera.width(), self.lbl_test_camera.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            elif current_tab == 3 and hasattr(self, 'lbl_auto_camera'):
                self.lbl_auto_camera.setPixmap(scaled_img.scaled(self.lbl_auto_camera.width(), self.lbl_auto_camera.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
        except Exception as e:
            traceback.print_exc()
            print(f"❌ [카메라 프레임 처리 오류] {e}", flush=True)
        finally:
            self.is_processing_frame = False

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; font-family: 'NanumGothic', sans-serif; }
            QTabWidget::pane { border: 2px solid #3a3a3a; background: #2b2b2b; border-radius: 10px; }
            QTabBar::tab { background: #3a3a3a; color: white; padding: 10px 10px; font-size: 18px; font-weight: bold; border-top-left-radius: 10px; border-top-right-radius: 10px; margin-right: 2px; min-width: 110px; }
            QTabBar::tab:selected { background: #4CAF50; color: white; }
            QTabBar::scroller { width: 0px; } 
            QLabel { color: #cccccc; font-size: 20px; }
            QListWidget { background-color: #3a3a3a; color: white; font-size: 20px; padding: 10px; border-radius: 10px; border: none;}
            QListWidget::item { padding: 15px; border-bottom: 1px solid #555; }
            QListWidget::item:selected { background-color: #4CAF50; border-radius: 5px; color: white; font-weight: bold; }
            QScrollBar:vertical { background: #2b2b2b; width: 15px; border-radius: 5px; }
            QScrollBar::handle:vertical { background: #555; border-radius: 5px; }
        """)

    def closeEvent(self, event):
        self.timer.stop()
        self.stop_tts_audio()
        if self.cap and self.cap.isOpened(): self.cap.release()
        event.accept()

if __name__ == '__main__':
    if "DISPLAY" not in os.environ: os.environ["DISPLAY"] = ":0"
    os.environ["QT_X11_NO_MITSHM"] = "1"
    app = QApplication(sys.argv)
    ex = SignLanguageApp()
    ex.show()
    sys.exit(app.exec_())
