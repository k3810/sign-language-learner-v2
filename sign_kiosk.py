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
import threading
from datetime import datetime
import numpy as np
import mediapipe as mp

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTabWidget, QLabel, 
                             QListWidget, QProgressBar, QStackedWidget, QSizePolicy, QMessageBox, QDialog)
from PyQt5.QtCore import Qt, QUrl, QTimer, QMetaObject, Q_ARG
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QPixmap, QImage

try:
    import google.generativeai as genai
    api_key = "AQ.Ab8RN6J3ygX4hk46pGZAKP5ZZ43Za79VO5tESQM_eov_cCyO4g"
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    GEMINI_READY = True
except ImportError:
    GEMINI_READY = False

class QuizResultDialog(QDialog):
    def __init__(self, correct_count, total_count, parent=None):
        super().__init__(parent)
        self.setWindowTitle("10초 챌린지 퀴즈 결과 분석")
        self.setFixedSize(500, 350)
        self.setStyleSheet("background-color: #2b2b2b; border-radius: 15px; font-family: 'NanumGothic', sans-serif;")
        
        layout = QVBoxLayout()
        lbl_title = QLabel("최종 성적표")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("color: #ffb74d; font-size: 32px; font-weight: bold; margin-top: 10px;")
        
        score_percent = int((correct_count / total_count) * 100)
        lbl_score = QLabel(f"{score_percent} 점")
        lbl_score.setAlignment(Qt.AlignCenter)
        
        if score_percent >= 80: color, msg = "#4CAF50", "최고의 실력입니다! 완벽해요!"
        elif score_percent >= 40: color, msg = "#FFD700", "조금만 더 연습하면 완벽해질 거예요!"
        else: color, msg = "#f44336", "학습 모드에서 다시 연습해 봅시다!"
        
        lbl_score.setStyleSheet(f"color: {color}; font-size: 70px; font-weight: bold; margin: 10px 0px;")
        lbl_detail = QLabel(f"총 {total_count}문제 중 {correct_count}문제를 통과했습니다.\n\n{msg}")
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
        self.setWindowTitle("수어 학습기 - 실전 마스터 에디션")
        self.resize(1024, 600)
        self.is_fullscreen = True
        self.showFullScreen()
        
        # [수정 완료] Holistic을 가볍고 빠른 Pose 모델로 교체 (엣지 최적화)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # 라즈베리파이 성능 향상을 위한 최저 복잡도
            min_detection_confidence=0.3, 
            min_tracking_confidence=0.3
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.base_path = "/home/a/Sign_Kiosk/media/수어 영상 mp4"
        self.default_expert_img_path = "/home/a/Sign_Kiosk/media/전문가 시연 이미지.png"
        self.csv_path = "/home/a/Sign_Kiosk/student_scores.csv"
        
        self.initial_load = True
        self.is_checking_motion = False 
        self.current_target_word = ""
        self.quiz_state = "IDLE" 
        self.quiz_target_word = ""
        self.quiz_start_time = 0
        self.quiz_current_q = 0
        self.quiz_correct_count = 0
        self.auto_detected_words = [] 
        self.last_auto_time = 0
        self.is_processing_frame = False

        self.words = ["감사합니다", "괜찮다"]
        self.word_data = {
            "감사합니다": "오른 손날을 세워 왼 손등을 가볍게 두 번 두드린다.",
            "괜찮다": "오른손을 펴서 손날을 턱에 가볍게 댄다."
        }
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        top_bar = QHBoxLayout()
        self.btn_window = QPushButton("창 모드")
        self.btn_fullscreen = QPushButton("전체화면")
        self.btn_exit = QPushButton("종료하기")
        
        btn_style = "background-color: #008CBA; color: white; padding: 8px 15px; font-size: 14px; font-weight: bold; border-radius: 5px;"
        self.btn_window.setStyleSheet(btn_style)
        self.btn_fullscreen.setStyleSheet(btn_style)
        self.btn_exit.setStyleSheet("background-color: #f44336; color: white; padding: 8px 15px; font-size: 14px; font-weight: bold; border-radius: 5px;")
        
        self.btn_window.clicked.connect(self.showNormal)
        self.btn_fullscreen.clicked.connect(self.showFullScreen)
        self.btn_exit.clicked.connect(self.closeEvent_manual)

        top_bar.addStretch()
        top_bar.addWidget(self.btn_window)
        top_bar.addWidget(self.btn_fullscreen)
        top_bar.addWidget(self.btn_exit)
        layout.addLayout(top_bar)

        self.tabs = QTabWidget()
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
        
        if self.words: self.change_word_by_index(0)

        self.cap = None
        for i in range(3):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                ret, _ = cap.read()
                if ret: 
                    self.cap = cap
                    break
                else: 
                    cap.release()
            else: 
                cap.release()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_frame)
        self.timer.start(35) 

    def create_word_selection_panel(self, list_widget):
        panel = QVBoxLayout()
        lbl_category = QLabel("일상생활 용어")
        lbl_category.setStyleSheet("font-size: 20px; font-weight: bold; color: #4CAF50;")
        
        list_widget.setFixedWidth(220)
        list_widget.addItems(self.words)
        list_widget.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        list_widget.itemClicked.connect(self.on_item_clicked)
        list_widget.currentRowChanged.connect(self.change_word_by_index)
        
        btn_layout = QHBoxLayout()
        btn_up = QPushButton("위로")
        btn_down = QPushButton("아래로")
        arrow_style = "background-color: #555; color: white; font-size: 16px; padding: 10px; border-radius: 5px; font-weight: bold;"
        btn_up.setStyleSheet(arrow_style)
        btn_down.setStyleSheet(arrow_style)
        btn_up.clicked.connect(lambda: list_widget.verticalScrollBar().setValue(list_widget.verticalScrollBar().value() - 50))
        btn_down.clicked.connect(lambda: list_widget.verticalScrollBar().setValue(list_widget.verticalScrollBar().value() + 50))
        
        btn_layout.addWidget(btn_up)
        btn_layout.addWidget(btn_down)
        panel.addWidget(lbl_category)
        panel.addWidget(list_widget)
        panel.addLayout(btn_layout)
        return panel

    def on_tab_changed(self, index):
        if hasattr(self, 'learn_media_player'): 
            self.learn_media_player.pause()
        self.auto_detected_words.clear()

    def on_item_clicked(self, item):
        word = item.text()
        if self.learn_list.currentItem() and self.learn_list.currentItem().text() == word:
            if self.tabs.currentIndex() == 0: 
                self.replay_video()
        else: 
            self.change_word(word)

    def change_word_by_index(self, index):
        if index < 0 or index >= len(self.words): return
        self.change_word(self.words[index])

    def change_word(self, word):
        self.current_target_word = word
        self.is_checking_motion = False
        self.practice_progress.setValue(0)
        self.practice_progress.setFormat("동작 정확도: 대기 중 (확인 버튼 클릭)")
        self.lbl_practice_camera.clear()
        self.lbl_practice_camera.setText("카메라 대기 중...\n\n('내 동작 확인하기'를 누르세요)")
        
        self.learn_list.blockSignals(True)
        self.practice_list.blockSignals(True)
        
        items = self.learn_list.findItems(word, Qt.MatchExactly)
        if items: self.learn_list.setCurrentItem(items[0])
        
        items_prac = self.practice_list.findItems(word, Qt.MatchExactly)
        if items_prac: self.practice_list.setCurrentItem(items_prac[0])
        
        self.learn_list.blockSignals(False)
        self.practice_list.blockSignals(False)
        self.update_media_and_text(word)

    def setup_learn_tab(self):
        layout = QHBoxLayout()
        self.learn_list = QListWidget()
        left_panel = self.create_word_selection_panel(self.learn_list)
        
        right_panel = QVBoxLayout()
        top_half = QVBoxLayout()
        lbl_video_title = QLabel("전문가 시연 영상")
        lbl_video_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #ffb74d;")
        
        self.learn_video_stack = QStackedWidget()
        self.learn_default_img = QLabel()
        self.learn_default_img.setAlignment(Qt.AlignCenter)
        self.learn_default_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        if os.path.exists(self.default_expert_img_path):
            self.learn_default_img.setPixmap(QPixmap(self.default_expert_img_path).scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
        self.learn_video_widget = QVideoWidget()
        self.learn_video_widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
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
        
        self.lbl_learn_image = QLabel("수형 사진 (파일 없음)")
        self.lbl_learn_image.setStyleSheet("background-color: transparent;")
        self.lbl_learn_image.setAlignment(Qt.AlignCenter)
        self.lbl_learn_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.lbl_learn_desc = QLabel("단어를 선택해 주세요.")
        self.lbl_learn_desc.setWordWrap(True)
        self.lbl_learn_desc.setStyleSheet("font-size: 20px; color: #FFD700; font-weight: bold; padding-top: 5px;")
        self.lbl_learn_desc.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.lbl_learn_desc.setMinimumHeight(40)
        
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
        top_half = QVBoxLayout()
        lbl_cam_title = QLabel("내 동작 확인 (거울 모드)")
        lbl_cam_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #008CBA;")
        
        self.lbl_practice_camera = QLabel("카메라 대기 중...\n\n('내 동작 확인하기'를 누르세요)")
        self.lbl_practice_camera.setAlignment(Qt.AlignCenter)
        self.lbl_practice_camera.setStyleSheet("background-color: black; color: white; font-size: 20px; border-radius: 10px;")
        self.lbl_practice_camera.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.practice_progress = QProgressBar()
        self.practice_progress.setRange(0, 100)
        self.practice_progress.setValue(0)
        self.practice_progress.setFormat("동작 정확도: 대기 중 (버튼을 누르세요)")
        self.practice_progress.setFixedHeight(40)
        self.practice_progress.setStyleSheet("""
            QProgressBar { border: 2px solid grey; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; color: white; background-color: #2b2b2b;}
            QProgressBar::chunk { background-color: #4CAF50; border-radius: 8px;}
        """)
        
        self.btn_check_motion = QPushButton("내 동작 확인하기")
        self.btn_check_motion.setStyleSheet("background-color: #4CAF50; color: white; font-size: 18px; font-weight: bold; padding: 15px; border-radius: 8px;")
        self.btn_check_motion.clicked.connect(self.start_motion_check)
        
        top_half.addWidget(lbl_cam_title)
        top_half.addWidget(self.lbl_practice_camera, stretch=1)
        top_half.addWidget(self.practice_progress)
        top_half.addWidget(self.btn_check_motion)
        
        bottom_half = QVBoxLayout()
        lbl_img_title = QLabel("수형 이미지 및 설명")
        lbl_img_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #4CAF50;")
        
        self.lbl_practice_image = QLabel("수형 사진")
        self.lbl_practice_image.setStyleSheet("background-color: transparent;")
        self.lbl_practice_image.setAlignment(Qt.AlignCenter)
        self.lbl_practice_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        self.lbl_practice_desc = QLabel("단어를 선택해 주세요.")
        self.lbl_practice_desc.setWordWrap(True)
        self.lbl_practice_desc.setStyleSheet("font-size: 20px; color: #FFD700; font-weight: bold; padding-top: 5px;")
        self.lbl_practice_desc.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        
        bottom_half.addWidget(lbl_img_title)
        bottom_half.addWidget(self.lbl_practice_image, stretch=1)
        bottom_half.addWidget(self.lbl_practice_desc)
        
        right_panel.addLayout(top_half, 6)
        right_panel.addLayout(bottom_half, 4)
        
        tab_layout.addLayout(left_panel, 1) 
        tab_layout.addLayout(right_panel, 3) 
        self.tab_practice.setLayout(tab_layout)

    def setup_test_tab(self):
        layout = QVBoxLayout()
        self.lbl_quiz_title = QLabel("10초 챌린지 퀴즈 (5문제)")
        self.lbl_quiz_title.setStyleSheet("font-size: 28px; font-weight: bold; color: #ffb74d;")
        self.lbl_quiz_title.setAlignment(Qt.AlignCenter)
        
        self.lbl_quiz_target = QLabel("아래 시작 버튼을 누르세요")
        self.lbl_quiz_target.setStyleSheet("font-size: 40px; font-weight: bold; color: white; background-color: #3a3a3a; padding: 20px; border-radius: 15px;")
        self.lbl_quiz_target.setAlignment(Qt.AlignCenter)
        
        self.lbl_quiz_timer = QLabel("0.0 초")
        self.lbl_quiz_timer.setStyleSheet("font-size: 30px; font-weight: bold; color: #f44336;")
        self.lbl_quiz_timer.setAlignment(Qt.AlignCenter)
        
        self.lbl_quiz_result = QLabel("")
        self.lbl_quiz_result.setAlignment(Qt.AlignCenter)
        self.lbl_quiz_result.setStyleSheet("font-size: 24px; font-weight: bold;")
        
        self.lbl_test_camera = QLabel("카메라 대기 중...\n\n(퀴즈가 시작되면 켜집니다)")
        self.lbl_test_camera.setAlignment(Qt.AlignCenter)
        self.lbl_test_camera.setStyleSheet("background-color: black; color: white; font-size: 20px; border-radius: 10px;")
        self.lbl_test_camera.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        btn_quiz_layout = QHBoxLayout()
        self.btn_start_quiz = QPushButton("퀴즈 시작 (1/5)")
        self.btn_start_quiz.setStyleSheet("background-color: #4CAF50; color: white; font-size: 20px; font-weight: bold; padding: 15px; border-radius: 10px;")
        self.btn_start_quiz.clicked.connect(self.next_quiz)
        
        self.btn_quiz_report = QPushButton("결과 분석")
        self.btn_quiz_report.setStyleSheet("background-color: #9C27B0; color: white; font-size: 20px; font-weight: bold; padding: 15px; border-radius: 10px;")
        self.btn_quiz_report.clicked.connect(self.show_quiz_result)
        self.btn_quiz_report.setEnabled(False)
        
        btn_quiz_layout.addWidget(self.btn_start_quiz)
        btn_quiz_layout.addWidget(self.btn_quiz_report)
        
        layout.addWidget(self.lbl_quiz_title)
        layout.addWidget(self.lbl_quiz_target)
        layout.addWidget(self.lbl_quiz_timer)
        layout.addWidget(self.lbl_quiz_result)
        layout.addWidget(self.lbl_test_camera, stretch=1)
        layout.addLayout(btn_quiz_layout)
        self.tab_test.setLayout(layout)

    def setup_auto_tab(self):
        layout = QVBoxLayout()
        cam_panel = QVBoxLayout()
        lbl_auto_cam = QLabel("실시간 동작 인식")
        lbl_auto_cam.setStyleSheet("font-size: 20px; font-weight: bold; color: #008CBA;")
        
        self.lbl_auto_camera = QLabel("카메라 구동 중...")
        self.lbl_auto_camera.setAlignment(Qt.AlignCenter)
        self.lbl_auto_camera.setStyleSheet("background-color: black; color: white; font-size: 20px; border-radius: 10px;")
        self.lbl_auto_camera.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        cam_panel.addWidget(lbl_auto_cam)
        cam_panel.addWidget(self.lbl_auto_camera, stretch=1)
        
        bottom_panel = QHBoxLayout()
        local_panel = QVBoxLayout()
        lbl_local_title = QLabel("기기 자체 인식 결과 (로컬)")
        lbl_local_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")
        
        self.lbl_local_result = QLabel("대기 중...")
        self.lbl_local_result.setAlignment(Qt.AlignCenter)
        self.lbl_local_result.setStyleSheet("font-size: 32px; font-weight: bold; color: white; background-color: #2b2b2b; border-radius: 10px; padding: 20px;")
        
        local_panel.addWidget(lbl_local_title)
        local_panel.addWidget(self.lbl_local_result, stretch=1)
        
        gemini_panel = QVBoxLayout()
        self.btn_gemini_translate = QPushButton("구글 AI 번역 시작")
        self.btn_gemini_translate.setStyleSheet("background-color: #9C27B0; color: white; font-size: 22px; font-weight: bold; padding: 15px; border-radius: 10px;")
        self.btn_gemini_translate.clicked.connect(self.run_gemini_translation)
        
        self.lbl_gemini_result = QLabel("구글 AI 번역 결과:\n대기 중...")
        self.lbl_gemini_result.setWordWrap(True)
        self.lbl_gemini_result.setStyleSheet("font-size: 24px; color: #FFD700; background-color: #3a3a3a; padding: 15px; border-radius: 10px;")
        self.lbl_gemini_result.setAlignment(Qt.AlignCenter)
        
        gemini_panel.addWidget(self.btn_gemini_translate)
        gemini_panel.addWidget(self.lbl_gemini_result, stretch=1)
        
        bottom_panel.addLayout(local_panel, 1)
        bottom_panel.addLayout(gemini_panel, 1)
        
        layout.addLayout(cam_panel, 6)
        layout.addLayout(bottom_panel, 4)
        self.tab_auto.setLayout(layout)

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.EndOfMedia:
            self.learn_video_stack.setCurrentIndex(0)

    def play_video(self):
        if hasattr(self, 'current_video_path') and self.current_video_path:
            safe_video_path = "/tmp/temp_sign_video.mp4"
            shutil.copy2(self.current_video_path, safe_video_path)
            media_content = QMediaContent(QUrl.fromLocalFile(safe_video_path))
            self.learn_media_player.setMedia(media_content)
            self.learn_video_stack.setCurrentIndex(1)
            self.learn_media_player.play()

    def replay_video(self):
        self.play_video()

    def update_media_and_text(self, word):
        desc = self.word_data.get(word, f"'{word}' 동작에 대한 상세 설명이 제공됩니다.")
        self.lbl_learn_desc.setText(desc)
        self.lbl_practice_desc.setText(desc)
        
        target_video = None
        target_img = None
        if os.path.exists(self.base_path):
            for file in os.listdir(self.base_path):
                clean_file = file.strip().lower()
                clean_word = word.replace(" ", "").lower()
                clean_file_no_space = clean_file.replace(" ", "")
                if clean_word in clean_file_no_space:
                    if clean_file.endswith(('.mp4', '.avi', '.mkv', '.mov')): target_video = os.path.join(self.base_path, file)
                    elif clean_file.endswith(('.jpg', '.jpeg', '.png')): target_img = os.path.join(self.base_path, file)
                    
        if target_img:
            safe_img_path = "/tmp/temp_sign_img.jpg"
            shutil.copy2(target_img, safe_img_path)
            pixmap = QPixmap(safe_img_path).scaled(450, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_learn_image.setPixmap(pixmap)
            self.lbl_practice_image.setPixmap(pixmap)
        else:
            no_img_style = "background-color: #3a3a3a; border-radius: 10px; color: #ff4444; font-size: 16px;"
            self.lbl_learn_image.setStyleSheet(no_img_style)
            self.lbl_learn_image.setText("사진 없음")
            self.lbl_practice_image.setStyleSheet(no_img_style)
            self.lbl_practice_image.setText("사진 없음")
            
        self.current_video_path = target_video
        if target_video and self.tabs.currentIndex() == 0:
            if self.initial_load:
                self.initial_load = False
                self.learn_video_stack.setCurrentIndex(0)
            else: 
                self.play_video()
        else:
            self.learn_media_player.stop()
            self.learn_video_stack.setCurrentIndex(0)

    def next_quiz(self):
        if self.quiz_current_q >= 5:
            self.quiz_state = "IDLE"
            self.btn_start_quiz.setText("모든 퀴즈 완료")
            self.btn_start_quiz.setEnabled(False)
            self.btn_quiz_report.setEnabled(True)
            self.lbl_test_camera.clear()
            self.lbl_test_camera.setText("카메라 대기 중...\n\n(퀴즈가 시작되면 켜집니다)")
            return
            
        self.quiz_current_q += 1
        self.quiz_target_word = random.choice(self.words)
        self.lbl_quiz_target.setText(f"문제 {self.quiz_current_q}/5: [{self.quiz_target_word}]")
        self.lbl_quiz_target.setStyleSheet("font-size: 36px; font-weight: bold; color: #FFD700; background-color: #2b2b2b; padding: 20px; border-radius: 15px;")
        
        self.lbl_quiz_result.setText("동작을 취하세요!")
        self.lbl_quiz_result.setStyleSheet("color: white;")
        self.quiz_start_time = cv2.getTickCount() / cv2.getTickFrequency()
        self.quiz_state = "PLAYING"
        self.btn_start_quiz.setEnabled(False)
        self.btn_quiz_report.setEnabled(False)

    def show_quiz_result(self):
        dialog = QuizResultDialog(self.quiz_correct_count, 5, self)
        dialog.exec_()
        self.lbl_quiz_target.setText(f"최종 성적: {self.quiz_correct_count}/5 통과")
        self.lbl_quiz_target.setStyleSheet("font-size: 36px; font-weight: bold; color: white; background-color: #9C27B0; padding: 20px; border-radius: 15px;")
        
        self.quiz_current_q = 0
        self.quiz_correct_count = 0
        self.btn_start_quiz.setText("다시 도전하기")
        self.btn_start_quiz.setEnabled(True)
        self.btn_quiz_report.setEnabled(False)
        self.lbl_test_camera.clear()
        self.lbl_test_camera.setText("카메라 대기 중...\n\n(퀴즈가 시작되면 켜집니다)")

    def log_score_to_csv(self, word, result, score):
        try:
            file_exists = os.path.isfile(self.csv_path)
            with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists: writer.writerow(["Date", "Time", "Target_Word", "Result", "Score(%)"])
                now = datetime.now()
                writer.writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), word, result, score])
        except Exception as e: pass

    def run_gemini_translation(self):
        if not self.auto_detected_words:
            self.lbl_gemini_result.setText("경고: 아직 인식된 동작이 없습니다.")
            return
        if not GEMINI_READY:
            self.lbl_gemini_result.setText("경고: Gemini API 설정 오류")
            return
            
        self.lbl_gemini_result.setText("구글 AI 번역 중...")
        words_str = ", ".join(self.auto_detected_words)
        
        def api_call():
            try:
                prompt = f"너는 수어 통역사야. 사용자가 다음 단어들을 순서대로 동작했어: [{words_str}]. 자연스러운 한국어 문장 딱 1개만 출력해. 부연 설명 금지."
                response = gemini_model.generate_content(prompt)
                # [버그 픽스] 스레드에서 GUI 직접 조작 시 발생하는 X11 Crash 방지
                QMetaObject.invokeMethod(self.lbl_gemini_result, "setText", 
                                         Qt.QueuedConnection, 
                                         Q_ARG(str, f"구글 AI 번역 결과:\n{response.text.strip()}"))
                self.auto_detected_words.clear() 
            except Exception as e:
                QMetaObject.invokeMethod(self.lbl_gemini_result, "setText", 
                                         Qt.QueuedConnection, 
                                         Q_ARG(str, f"오류: 네트워크 또는 API 오류 발생\n{e}"))
                                         
        threading.Thread(target=api_call, daemon=True).start()

    def start_motion_check(self):
        self.is_checking_motion = True
        self.practice_progress.setFormat("동작 정확도: %p%")
        self.practice_progress.setValue(0)

    def compute_pose_score(self, pose_landmarks, target_word):
        if not pose_landmarks: return 0
        try:
            nose = np.array([pose_landmarks.landmark[0].x, pose_landmarks.landmark[0].y])
            l_shoulder = np.array([pose_landmarks.landmark[11].x, pose_landmarks.landmark[11].y])
            r_shoulder = np.array([pose_landmarks.landmark[12].x, pose_landmarks.landmark[12].y])
            l_wrist = np.array([pose_landmarks.landmark[15].x, pose_landmarks.landmark[15].y])
            r_wrist = np.array([pose_landmarks.landmark[16].x, pose_landmarks.landmark[16].y])
        except Exception: return 0
            
        shoulder_width = np.linalg.norm(l_shoulder - r_shoulder) + 1e-6
        score = 0
        if target_word == "감사합니다":
            hands_dist = np.linalg.norm(r_wrist - l_wrist) / shoulder_width
            if hands_dist < 0.6: score = 100 - int(hands_dist * 120)
        elif target_word == "괜찮다":
            nose_to_hand = np.linalg.norm(r_wrist - nose) / shoulder_width
            if nose_to_hand < 0.8: score = 100 - int(nose_to_hand * 100)
            
        return max(0, min(98, score))

    def update_camera_frame(self):
        if getattr(self, 'is_processing_frame', False): return
        self.is_processing_frame = True
        
        if self.cap is None or not self.cap.isOpened(): 
            self.is_processing_frame = False
            return
        
        current_tab = self.tabs.currentIndex()
        needs_processing = False
        if current_tab == 1 and self.is_checking_motion: needs_processing = True
        elif current_tab == 2 and self.quiz_state == "PLAYING": needs_processing = True
        elif current_tab == 3: needs_processing = True

        if not needs_processing:
            self.cap.grab()
            self.is_processing_frame = False
            return

        ret, frame = self.cap.read()
        if not ret: 
            self.is_processing_frame = False
            return
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, _ = rgb_frame.shape
        box_margin_x = int(w * 0.15)
        box_margin_y = int(h * 0.1)
        cv2.rectangle(rgb_frame, (box_margin_x, box_margin_y), (w - box_margin_x, h - box_margin_y), (0, 255, 0), 2)
        
        # [수정 완료] 프로세스 실행 부분 (Pose 모델로 연산 최적화)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks: 
            self.mp_drawing.draw_landmarks(
                rgb_frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS, 
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=3, circle_radius=4), 
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=3, circle_radius=2)
            )
        
        score = 0
        detected_word = "대기 중"
        
        if results.pose_landmarks:
            if current_tab == 1 and self.is_checking_motion: 
                score = self.compute_pose_score(results.pose_landmarks, self.current_target_word)
            elif current_tab == 2 and self.quiz_state == "PLAYING": 
                score = self.compute_pose_score(results.pose_landmarks, self.quiz_target_word)
            elif current_tab == 3:
                score_thx = self.compute_pose_score(results.pose_landmarks, "감사합니다")
                score_ok = self.compute_pose_score(results.pose_landmarks, "괜찮다")
                if score_thx > 80: detected_word, score = "감사합니다", score_thx
                elif score_ok > 80: detected_word, score = "괜찮다", score_ok

        if current_tab == 1 and self.is_checking_motion:
            current_val = self.practice_progress.value()
            new_val = int(current_val * 0.8 + score * 0.2)
            self.practice_progress.setValue(new_val)

        if current_tab == 2 and self.quiz_state == "PLAYING":
            elapsed = (cv2.getTickCount() / cv2.getTickFrequency()) - self.quiz_start_time
            rem_time = max(0, 10.0 - elapsed)
            self.lbl_quiz_timer.setText(f"{rem_time:.1f} 초")
            if score > 85: 
                self.quiz_state = "SUCCESS"
                self.quiz_correct_count += 1
                self.lbl_quiz_result.setText("완벽합니다! (통과)")
                self.lbl_quiz_result.setStyleSheet("font-size: 30px; font-weight: bold; color: #4CAF50;")
                self.btn_start_quiz.setText(f"다음 퀴즈 ({self.quiz_current_q + 1}/5)")
                self.btn_start_quiz.setEnabled(True)
                self.lbl_test_camera.clear()
                self.lbl_test_camera.setText("카메라 대기 중...\n\n(퀴즈가 시작되면 켜집니다)")
            elif rem_time == 0:
                self.quiz_state = "FAIL"
                self.lbl_quiz_result.setText("아쉽습니다 (시간 초과)")
                self.lbl_quiz_result.setStyleSheet("font-size: 30px; font-weight: bold; color: #f44336;")
                self.btn_start_quiz.setText(f"다음 퀴즈 ({self.quiz_current_q + 1}/5)")
                self.btn_start_quiz.setEnabled(True)
                self.lbl_test_camera.clear()
                self.lbl_test_camera.setText("카메라 대기 중...\n\n(퀴즈가 시작되면 켜집니다)")
                if self.quiz_current_q >= 5:
                    self.btn_start_quiz.setText("모든 퀴즈 완료")
                    self.btn_start_quiz.setEnabled(False)
                    self.btn_quiz_report.setEnabled(True)

        if current_tab == 3: 
            if score > 80 and detected_word != "대기 중":
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                if current_time - self.last_auto_time > 2.0: 
                    self.auto_detected_words.append(detected_word)
                    self.lbl_local_result.setText(f"[{detected_word}]")
                    self.last_auto_time = current_time

        q_img = QImage(rgb_frame.data, w, h, rgb_frame.strides[0], QImage.Format_RGB888)
        scaled_img = QPixmap.fromImage(q_img)
        
        if current_tab == 1:
            cw, ch = self.lbl_practice_camera.width(), self.lbl_practice_camera.height()
            if cw > 0 and ch > 0: self.lbl_practice_camera.setPixmap(scaled_img.scaled(cw, ch, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif current_tab == 2:
            cw, ch = self.lbl_test_camera.width(), self.lbl_test_camera.height()
            if cw > 0 and ch > 0: self.lbl_test_camera.setPixmap(scaled_img.scaled(cw, ch, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif current_tab == 3:
            cw, ch = self.lbl_auto_camera.width(), self.lbl_auto_camera.height()
            if cw > 0 and ch > 0: self.lbl_auto_camera.setPixmap(scaled_img.scaled(cw, ch, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.is_processing_frame = False

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; font-family: 'NanumGothic', 'Malgun Gothic', sans-serif; }
            QTabWidget::pane { border: 2px solid #3a3a3a; background: #2b2b2b; border-radius: 10px; }
            QTabBar::tab { background: #3a3a3a; color: white; padding: 10px 20px; font-size: 18px; font-weight: bold; border-top-left-radius: 10px; border-top-right-radius: 10px; margin-right: 5px; min-width: 140px; }
            QTabBar::tab:selected { background: #4CAF50; color: white; }
            QLabel { color: #cccccc; font-size: 20px; }
            QListWidget { background-color: #3a3a3a; color: white; font-size: 20px; padding: 10px; border-radius: 10px; border: none;}
            QListWidget::item { padding: 15px; border-bottom: 1px solid #555; }
            QListWidget::item:selected { background-color: #4CAF50; border-radius: 5px; color: white; font-weight: bold; }
            QScrollBar:vertical { background: #2b2b2b; width: 15px; border-radius: 5px; }
            QScrollBar::handle:vertical { background: #555; border-radius: 5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)

    def closeEvent_manual(self): self.close()
    
    def closeEvent(self, event):
        self.timer.stop()
        if self.cap and self.cap.isOpened(): self.cap.release()
        event.accept()

if __name__ == '__main__':
    if "DISPLAY" not in os.environ: os.environ["DISPLAY"] = ":0"
    os.environ["QT_X11_NO_MITSHM"] = "1"
    app = QApplication(sys.argv)
    ex = SignLanguageApp()
    ex.show()
    sys.exit(app.exec_())
