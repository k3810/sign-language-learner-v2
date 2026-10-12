# 🏭 하이브리드 산업 안전 수어 교육 키오스크 (Customizable Edge-AI Sign Language Learner)

**'전국마이스터고 스타프로젝트 AI Competition' 출품작 (팀명: 밥먹고 AI만)**

> **⚠️ [필수 안내] 심사위원 및 운영진 대용량 파일 다운로드 링크**
> 깃허브 용량 제한(300MB) 규정에 따라, NPU 구동을 위한 AI 모델 가중치 파일(.hef)과 시연 영상 등 대용량 미디어 파일은 아래 구글 드라이브에 분리하여 업로드하였습니다. 코드를 실행하시기 전에 아래 링크에서 파일을 다운로드하여 프로젝트 경로에 포함해 주시기 바랍니다.
> 👉 **[대용량 파일 다운로드 링크 (Google Drive)](https://drive.google.com/drive/folders/1fjnOGX_P6aI3tNnjpgisrnIRt4fs9pDQ?usp=drive_link)**
>
> 🎬 **[3분 핵심 데모 시연 영상 보기 (클릭)](https://youtu.be/GNS1Rb4az68))**
---

본 프로젝트는 청각장애인과 비장애인이 함께 안전하게 소통하며 일할 수 있도록 돕는 실시간 수어 학습 기기입니다. 엣지 컴퓨팅(Edge Computing)과 로컬 AI 서버를 결합하여, 학교 실습실이나 산업 현장에 당장 필요한 '맞춤형 안전 수어'를 관리자가 직접 3D 뼈대 데이터로 추가하고 학생들이 즉각적인 AI 피드백을 받으며 배울 수 있도록 개발되었습니다.

## 1. 문제정의

**"생명과 직결된 산업 현장, 그러나 '안전 수어'의 국가 표준은 없습니다."**

* **현장 맞춤형 수어의 부재:** 기계, 용접, 건설 등 산업 현장마다 사용하는 위험 장비가 다릅니다. 하지만 기존 수어 번역기들은 대중적인 일상 단어만 제공할 뿐, "프레스 기계 비상 정지"와 같은 특수한 생명 직결 단어는 알려주지 않습니다.
* **현장 은어와 소통 단절:** 표준 수어가 없기 때문에 공장마다 임의로 만든 '그들만의 은어(바디랭귀지)'를 사용하며, 노동자가 이직할 때마다 안전 교육을 처음부터 다시 받아야 하는 치명적인 사각지대가 발생합니다.
* **우리의 해결책 (대체 불가능성):** 세상의 모든 단어를 얕게 아는 클라우드 번역기가 아닙니다. 관리자가 우리 실습장에 딱 맞는 안전 수어를 직접 기계에 녹화(Teaching)해서 넣고, 학생들은 오프라인 엣지 환경에서 NPU와 로컬 LLM을 통해 정밀한 채점과 다정한 대화형 피드백을 받으며 완벽하게 훈련할 수 있는 '현장 맞춤형 수어 학습 플랫폼'입니다.

## 2. 아키텍처

본 시스템은 기기의 부하를 줄이고 지연 시간을 최소화하기 위해 **투트랙(Two-Track) 하이브리드 구조**와 6가지 모듈화된 기능을 채택했습니다.

* **엣지 컴퓨팅 (Raspberry Pi 5 + Hailo-10H NPU):** 
  웹캠 영상을 받아 `MediaPipe`로 손과 몸의 75차원 3D 뼈대 좌표를 추출하여 자체 구현한 DTW 알고리즘으로 시계열 동작 정확도를 채점합니다. 특히 '연습 모드'에서는 기기 내 NPU에서 시각-언어 모델(`Qwen2-VL-2B`)을 온디바이스로 구동하여 정밀한 시각적 자세 교정 피드백을 제공합니다.
* **로컬 AI 서버 (노트북 연동):** 
  '자율 모드'와 '대화 모드' 등 긴 문장 생성이 필요한 경우, 라즈베리파이가 1차 추출한 데이터를 동일 네트워크 상의 노트북으로 전송합니다. 노트북의 순수 텍스트 언어 모델(`Qwen2.5:3b`)이 맥락에 맞는 칭찬과 상세한 피드백을 생성하여 반환합니다.

* **핵심 기능 (6가지 모드):**
  1. **정답지 스튜디오:** 관리자가 직접 새로운 수어 단어의 3D 뼈대 데이터(JSON)와 시연 영상을 추가하는 레코딩 스튜디오.
  2. **학습 모드:** 전문가 시연 영상과 수형 설명을 통해 기본 동작 숙지.
  3. **연습 모드:** 사용자의 동작을 NPU 비전 모델과 DTW로 분석하여 채점 및 교정.
  4. **확인 모드:** 7가지 미니게임(객관식, 초성 퀴즈, 서바이벌 등)으로 지루함 없이 복습.
  5. **자율 모드:** 자유로운 수어 동작을 인식해 로컬 LLM이 다정한 문장으로 번역.
  6. **AI와 대화:** STT(음성 인식)와 카메라를 융합하여 AI 어시스턴트와 심화 학습 진행.

## 3. 📁 프로젝트 파일 구성 및 역할 (File Architecture)

본 프로젝트는 엣지 NPU 가속, 실시간 관절 추적, 관리자용 데이터 구축, 로컬 LLM 통신이 유기적으로 분업화된 구조로 설계되었습니다. 심사위원님들의 원활한 코드 리뷰를 위해 핵심 파일 27개의 역할을 명시합니다.

| 구분 | 파일명 | 역할 및 상세 설명 |
| :--- | :--- | :--- |
| **핵심 애플리케이션** | `a.py` | **메인 키오스크 실행기.** PyQt5 기반 GUI, MediaPipe 75차원 좌표 추출, DTW 알고리즘 실시간 수어 채점 |
| | `sign_kiosk.py` | 키오스크 시스템 코어 제어 로직 모듈 |
| | `sign_ui.py` | 키오스크 사용자 인터페이스(UI) 렌더링 및 화면 전환 핸들러 |
| | `sign_ui_hailo10.py` | Hailo-10H NPU 연동 전용 수어 학습 화면 모듈 |
| **정답지 스튜디오** | `truth_studio.py` | **관리자용 수어 녹화기.** 카메라 영상에서 3D 관절 데이터를 추출해 신규 단어(JSON)를 구축하는 도구 |
| **로컬 서버 & AI 추론** | `vlm_server.py` | 로컬 노트북 구동용 Flask API 서버 (Qwen2.5 언어 모델 기반 맞춤형 문장 및 피드백 생성) |
| | `test_vlm.py` | VLM API 서버 통신 및 응답 속도 검증용 테스트 스크립트 |
| | `patch_vlm.py` | 언어 모델 응답 파싱 및 포맷 정규화 패치 스크립트 |
| | `restore_pure_vlm.py` | VLM 모델 초기 환경 설정 및 복구 스크립트 |
| | `vlm_memory_fix.py` | 엣지 환경 메모리 누수 방지 및 버퍼 최적화 유틸리티 |
| **수어 데이터셋 (JSON)** | `dynamic_sign_model.json` | 산업 안전 특수 수어 10종을 포함한 시계열 3D 뼈대 정답지 좌표 템플릿 데이터 |
| | `dynamic_sign_desc.json` | 수어 단어별 텍스트 설명, 동작 포인트, 수형 안내 메타데이터 |
| | `dynamic_sign_category.json` | 수어 카테고리 분류(산업 안전, 기계 조작, 일상 등) 정의 데이터 |
| | `open_sign_model_v1.json` | 공공 수어사전 기반 초기 베이스라인 템플릿 모델 데이터 |
| **데이터 처리 파이프라인** | `extract_dynamic.py` | 실시간 카메라 프레임에서 동적 시계열 관절 벡터 추출 |
| | `extract_json.py` | 추출된 랜드마크 좌표를 표준 JSON 구조로 직렬화 |
| | `make_vector_json.py` | 관절 간 상대 거리 및 방향 벡터 정규화 변환 스크립트 |
| | `make_custom_json.py` | 사용자 정의 신규 수어 데이터 통합 빌더 |
| | `make_6d_json.py` | 6차원 핵심 특징점 추출 및 데이터 경량화 스크립트 |
| | `make_full_6d_json.py` | 전신 자세 좌표 결합 및 다차원 벡터 확장 변환기 |
| | `setup_metadata.py` | 학습 데이터 인덱싱 및 메타데이터 자동 동기화 도구 |
| **AI 모델 & 하드웨어** | `best_sign_model.onnx` | 경량화된 수어 분류 및 포즈 추정 ONNX 딥러닝 모델 가중치 |
| | `factory_reset_hailo.py` | Hailo-10H NPU 가속기 장치 드라이버 및 런타임 초기화 도구 |
| **실행 스크립트 (Shell)** | `run_kiosk.sh` | **통합 원클릭 런처.** NPU 백그라운드 프로세스 및 GUI 메인 앱 동시 기동 |
| | `run_truth_studio.sh` | 정답지 스튜디오 단독 기동용 단축 쉘 스크립트 |
| | `start_sign_kiosk.sh` | 시스템 데몬 초기화 및 메인 키오스크 실행 환경 셋업 스크립트 |
| | `start_ksl_kiosk.sh` | 한국수어(KSL) 학습 모드 보조 실행 스크립트 |

## 4. 사용 스택

* **Hardware:** Raspberry Pi 5 (8GB), Hailo-10H AI HAT, USB 웹캠, 로컬 서버용 노트북
* **Software (GUI & Logic):** Python 3, PyQt5, OpenCV, NumPy (가상 키보드용 커스텀 한글 오토마타 자체 개발)
* **AI & Vision:** MediaPipe (Holistic 3D 뼈대 추출)
* **ETC:** Edge-TTS / gTTS (음성 합성), SpeechRecognition (음성 인식), Flask (기기 간 비동기 통신)

## 5. 실행방법

> ⚠️ **주의:** 본 프로젝트는 라즈베리파이5 NPU 하드웨어와 노트북이 동일 네트워크에 연결된 환경에 최적화되어 있습니다.

```bash
# 1. 저장소 클론 (라즈베리파이 환경)
git clone [https://github.com/k3810/sign-language-learner-v2.git](https://github.com/k3810/sign-language-learner-v2.git)
cd sign-language-learner-v2

# 2. 대용량 파일 추가 (필수)
# 최상단의 '구글 드라이브 링크'에서 파일을 다운로드 받아 프로젝트 경로 내에 배치합니다.

# 3. 필수 패키지 설치
pip install -r requirements.txt

# 4. 노트북(서버)에서 LLM API 서버 실행
python vlm_server.py

# 5. 통합 키오스크 런처 실행 (NPU 백그라운드 서버 & GUI 자동 동시 기동)
bash run_kiosk.sh
```

## 6. 핵심 성과 및 실측 지표 (Do Things That Don't Scale)
자동 로그인 기능 개발에 매몰되지 않고, 초기 스타트업의 철학에 따라 현장에 기기를 먼저 설치한 뒤 아날로그 방식(타임스탬프 로그와 수기 관찰 명단 교차 대조)으로 직접 실측한 허수 없는 지표입니다.
* **주간 활성 사용자:** 20명
* **재방문율 (Retention):** 50% (1주 차 핵심 행동 완료자 14명 중 7명 자발적 재방문)
* **퍼널 개선:** 현장 관찰을 통해 대기 시간 연장(3초➡️5초) 및 UI 개선 후 핵심 행동 완료율 70% ➡️ 90% 달성

## 7. 듀얼 노드 분산 엣지 아키텍처 (한계 돌파)
단일 기기의 연산 한계를 넘기 위해 '듀얼 노드 분산 시스템'을 구축 중입니다. 무거운 비전 인식은 우분투 노드에 전담시키고, 가벼운 뼈대 데이터만 UDP 랜선 통신으로 데비안 메인 노드에 전송하여 엣지의 처리 속도와 안정성을 극대화했습니다. 

## 8. AI 윤리 및 생성형 AI 활용 내역 (100% AI 페어 프로그래밍)

본 프로젝트는 기획부터 완성에 이르기까지 현대적인 프롬프트 엔지니어링 및 QA 주도 개발 방식을 투명하게 적용했습니다.
* **사용 AI:** Google Gemini Pro
* **활용 방식 (우리의 AI 메인 개발자):** 프로젝트 문제 정의부터 기능 구성, UI 화면 배치까지 모든 아이디어를 팀원들이 구상하여 AI에게 프롬프트로 지시했습니다. AI가 작성한 코드를 라즈베리파이 실물 기기에 입력하여 구동하고, 터미널의 에러 로그를 캡처해 다시 AI에게 피드백하는 '무한 반복 디버깅(QA)' 과정을 거쳐 완성했습니다. 코딩 타자 치기는 AI가 전담하고, 기획과 실물 테스트는 팀원들이 전담하는 철저한 분업을 통해 프로젝트의 완성도와 속도를 극대화했습니다.

## 9. 서비스 구동 모델 및 오픈소스 라이선스

* **구동 AI 모델 및 라이브러리:**
  * **Google MediaPipe:** 사용자 포즈 및 골격 3D 추정
  * **Qwen2-VL:** Hailo-10H NPU 엣지 환경에서의 비전 분석
  * **Ollama (Qwen2.5):** 로컬 텍스트 생성 및 대화형 피드백
  * **Google Speech Recognition & Edge-TTS / gTTS:** 음성-텍스트 상호 변환 (STT/TTS)
* **오픈소스 패키지 라이선스 맵핑:**
  * **Apache 2.0 License:** OpenCV (영상 처리), MediaPipe (비전), Requests (API 통신)
  * **BSD 3-Clause License:** NumPy (행렬 연산), Flask (로컬 웹서버), SpeechRecognition (음성 인식)
  * **GPL v3 License:** PyQt5 (GUI 키오스크 화면), Edge-TTS (로컬 음성 합성)
  * **HPND License:** Pillow (PIL, 이미지 처리)
  * **Hailo Platform License:** `hailo_platform` (NPU 구동용 SDK)

## 10. 학습 데이터 및 미디어(이미지/폰트) 출처

* **수어 데이터 (공공):** 기기 내 기본 탑재된 일부 수어 사진 및 영상은 '국립국어원 한국수어사전' 공공 데이터를 활용했습니다. 명시된 **'저작자표시-비영리-변경금지(CC BY-NC-ND 2.0 KR)'** 조항을 엄격히 준수하여, 어떠한 영리적 목적 없이 원본 데이터의 가공(크롭, 필터, 편집 등)을 일절 배제한 채 있는 그대로 사용하고 출처를 명시했습니다.
* **수어 데이터 (자체 제작):** 그 외 산업 현장 맞춤형 수어 및 뼈대 추출 데이터는 본 프로젝트 팀원(마이스터고 학생)들이 현장 수요에 맞추어 스튜디오에서 직접 촬영하고 가공한 순수 창작물입니다.
* **폰트:** UI 렌더링에 사용된 '나눔고딕(NanumGothic)'은 무료 재배포가 가능한 오픈 폰트 라이선스(OFL)를 적용했습니다.

## 11. 프로젝트 라이선스

```text
MIT License

Copyright (c) 2026 k3810

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
