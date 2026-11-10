# 🪐 TESS 외계행성 탐지 프로젝트 (Exoplanet Transit Detection)

NASA **TESS**(Transiting Exoplanet Survey Satellite)가 관측한 별의 **밝기 시계열 데이터(Light Curve)** 를 분석해서,
행성이 별 앞을 지나갈 때 생기는 **Transit(밝기 감소) 패턴**을 찾아내고 외계행성 후보를 분류하는 프로젝트입니다.

> 이 문서 하나만 위에서부터 따라 하면 **환경 세팅 → 첫 실습 → GitHub 정리**까지 끝납니다.
> 명령어는 복사해서 **PowerShell**에 붙여넣으면 됩니다.

---

## 📦 0. 전체 그림 (한눈에 보기)

```txt
NASA TESS Data
  → Light Curve 수집        (lightkurve 라이브러리로 다운로드)
  → Python 전처리           (결측치 제거, 정규화)
  → Signal Processing       (트렌드 제거 / flatten)
  → Feature Engineering     (주기, 깊이, 폭 등 특징 추출)
  → ML / Deep Learning      (후보 분류)
  → Exoplanet Probability   (외계행성일 확률)
  → FastAPI                 (결과를 API로 제공)
  → Web Visualization       (그래프 시각화)
```

이 프로젝트는 위 흐름을 **단계(Step)별로 하나씩** 완성해 갑니다. 지금은 **Step 1~2(환경 세팅 + 첫 데이터 시각화)** 가 목표입니다.

---

## 🧰 1. 먼저 설치할 프로그램 (한 번만)

| 프로그램 | 용도 | 다운로드 |
|---|---|---|
| **Python 3.12** | 프로젝트 실행 언어 | https://www.python.org/downloads/release/python-3127/ |
| **VS Code** | 코드 편집기 | https://code.visualstudio.com/ |
| **Git** | 버전관리 / GitHub 업로드 | https://git-scm.com/download/win |
| **GitHub 계정** | 결과물 정리/공개 | https://github.com/ |

> ⚠️ **Python 버전 안내**
> 지금 PC에는 **Python 3.14**가 깔려 있는데, 천문/과학 라이브러리(`lightkurve`, `astropy`, `scipy`)는
> 아직 3.14에서 설치가 깨지는 경우가 많습니다. **가장 안정적인 3.12 버전**을 따로 설치하길 권장합니다.
> 설치할 때 **"Add python.exe to PATH"** 체크박스를 꼭 켜세요.
> (3.12로 설치하면 아래 명령어의 `py -3.12` 부분이 그 버전을 콕 집어서 사용합니다.)

설치 확인:

```powershell
py -3.12 --version
```

`Python 3.12.x` 가 나오면 성공입니다. (3.12를 안 깔았다면 `python --version` 으로 확인)

---

## 📁 2. 프로젝트 폴더 & 가상환경(venv) 만들기

가상환경은 이 프로젝트 전용 "독립된 파이썬 방"입니다. 다른 프로젝트와 라이브러리가 섞이지 않게 해줍니다.

PowerShell을 열고 아래를 **한 줄씩** 실행하세요.

**① 프로젝트 폴더로 이동**
```powershell
cd "C:\Users\조희선\Downloads\AI_Ethic_Monitoring--main\tess-exoplanet-detection"
```

**② 가상환경 생성** (3.12 기준)
```powershell
py -3.12 -m venv .venv
```

**③ 가상환경 켜기(활성화)**
```powershell
.\.venv\Scripts\Activate.ps1
```
> 줄 맨 앞에 `(.venv)` 가 보이면 성공입니다.
> 만약 *"실행 정책" 오류*가 나면 아래를 한 번 실행한 뒤 다시 ③을 하세요.
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

**④ pip 최신화**
```powershell
python -m pip install --upgrade pip
```

---

## 📚 3. 라이브러리 설치

이 폴더의 `requirements.txt` 에 필요한 라이브러리 목록이 들어 있습니다. 한 번에 설치합니다.

```powershell
pip install -r requirements.txt
```

설치되는 핵심 라이브러리:

| 라이브러리 | 역할 |
|---|---|
| `lightkurve` | **TESS/Kepler 데이터 다운로드 + light curve 처리** (이 프로젝트의 핵심) |
| `numpy`, `pandas` | 데이터 계산/표 처리 |
| `matplotlib` | 그래프 그리기 |
| `astropy` | 천문 데이터 포맷(FITS) 처리 |
| `scipy` | 신호처리/통계 |
| `scikit-learn` | 머신러닝 (후반부) |
| `jupyter` | 실험용 노트북 |

설치 확인:
```powershell
python -c "import lightkurve; print('lightkurve', lightkurve.__version__)"
```
버전이 출력되면 세팅 완료입니다. 🎉

---

## 🚀 4. 첫 실습 — TESS 별 하나의 Light Curve 그려보기

유명한 외계행성계 **Pi Mensae (TESS 첫 발견 행성 중 하나)** 의 데이터를 받아 그래프로 그려봅니다.

이 폴더의 `src\01_download_lightcurve.py` 를 실행하세요.

```powershell
python src\01_download_lightcurve.py
```

실행하면:
1. NASA 데이터 서버에서 Pi Mensae의 관측 데이터를 자동으로 내려받고,
2. 밝기 시계열 그래프를 화면에 띄우고,
3. `outputs\pi_men_lightcurve.png` 로 저장합니다.

그래프에서 **아래로 뾰족하게 떨어지는 지점**이 바로 행성이 별을 가린 **Transit**입니다. 이게 이 프로젝트의 출발점입니다.

> 처음 실행 시 데이터 다운로드로 1~2분 걸릴 수 있습니다. (NASA 서버에서 받아옴)

---

## 🐙 5. GitHub에 정리하기

결과물을 GitHub 저장소로 올려 포트폴리오로 만듭니다.

**① Git 사용자 정보 설정** (최초 1회만, 본인 정보로)
```powershell
git config --global user.name "본인이름"
git config --global user.email "본인GitHub이메일"
```

**② 이 폴더를 git 저장소로 만들기**
```powershell
cd "C:\Users\조희선\Downloads\AI_Ethic_Monitoring--main\tess-exoplanet-detection"
```
```powershell
git init
```

**③ 첫 커밋**
```powershell
git add .
```
```powershell
git commit -m "init: TESS 외계행성 탐지 프로젝트 세팅 + 첫 light curve 실습"
```

**④ GitHub에서 빈 저장소 만들기**
- https://github.com/new 접속
- Repository name: `tess-exoplanet-detection`
- **Public** 선택, README/.gitignore는 **체크하지 말고** (이미 로컬에 있음) → **Create repository**

**⑤ 원격 저장소 연결 & 업로드** (아래 URL의 `본인아이디` 교체)
```powershell
git remote add origin https://github.com/본인아이디/tess-exoplanet-detection.git
```
```powershell
git branch -M main
```
```powershell
git push -u origin main
```

> 이후에는 작업할 때마다 아래 3줄만 반복하면 GitHub에 반영됩니다.
> ```powershell
> git add .
> git commit -m "작업 내용 설명"
> git push
> ```

---

## 🗂️ 6. 폴더 구조

```txt
tess-exoplanet-detection/
├── README.md              ← 이 문서 (프로젝트 가이드)
├── requirements.txt       ← 설치할 라이브러리 목록
├── .gitignore             ← GitHub에 안 올릴 파일 목록
├── src/                   ← 실행 코드 (단계별)
│   ├── 01_download_lightcurve.py   ← Step 1: 데이터 받아 그리기
│   ├── 02_flatten_lightcurve.py    ← Step 2: flatten 전처리
│   ├── 03_find_period_bls.py       ← Step 3: BLS 주기 탐색
│   ├── 04_extract_features.py      ← Step 4: 특징 추출 → CSV
│   ├── 05_train_classifier.py      ← Step 5: 머신러닝 분류 (시뮬레이션 라벨)
│   ├── 05b_train_classifier_toi.py ← Step 5+: 진짜 라벨(NASA TOI) 분류
│   └── 06_api.py                   ← Step 6+7: FastAPI + 웹 대시보드
├── web/                   ← 웹 대시보드 프런트엔드
│   └── index.html
├── outputs/               ← 생성된 그래프/CSV 결과물
└── data/                  ← 다운로드한 원본 데이터 (git 제외)
```

---

## 🧭 7. 앞으로의 로드맵 (단계별)

| 단계 | 할 일 | 핵심 개념 | 코드 | 결과물 |
|---|---|---|---|---|
| ✅ **Step 1** | 환경 세팅 + 데이터 1개 받아 그리기 | lightkurve, light curve | `src/01_download_lightcurve.py` | `outputs/pi_men_lightcurve.png` |
| ✅ **Step 2** | 전처리 — 노이즈/트렌드 제거 (`flatten`) | 신호처리 기초 | `src/02_flatten_lightcurve.py` | `outputs/pi_men_flatten.png` |
| ✅ **Step 3** | Transit 주기 찾기 (BLS 알고리즘) | 주기 탐색, folding | `src/03_find_period_bls.py` | `outputs/pi_men_folded.png`, `outputs/pi_men_bls_periodogram.png` |
| ✅ **Step 4** | 여러 별 데이터 모으기 + 특징 추출 | Feature Engineering | `src/04_extract_features.py` | `outputs/features.csv` |
| ✅ **Step 5** | 머신러닝으로 "행성 후보 vs 아님" 분류 (개념 학습) | scikit-learn | `src/05_train_classifier.py` | `outputs/predictions.csv`, `outputs/ml_feature_space.png` |
| ✅ **Step 5+** | **진짜 라벨(NASA TOI)로 "확정행성 vs 가짜신호" 분류** | 실데이터 검증 | `src/05b_train_classifier_toi.py` | `outputs/toi_labeled.csv`, `outputs/toi_feature_importance.png` |
| ✅ **Step 6** | 결과를 FastAPI로 제공 | API 개발 | `src/06_api.py` | `/stars`, `/predict`, `/docs` |
| ✅ **Step 7** | 웹에서 그래프로 시각화 | 데이터 시각화 | `src/06_api.py` + `web/index.html` | `http://127.0.0.1:8000/` 대시보드 |

**🎉 로드맵 7단계 전체 완주!** NASA 광도 데이터 → 신호처리 → 특징공학 → 머신러닝 → API → 웹 시각화까지 이어지는 엔드투엔드 파이프라인이 완성되었습니다.

각 단계가 끝날 때마다 `git commit` + `git push` 로 GitHub에 쌓으면 그대로 포트폴리오가 됩니다.

### 🔬 Step 5+ — 시뮬레이션에서 진짜 데이터로

머신러닝 분류는 두 단계로 발전시켰습니다.

1. **개념 학습 (`05_train_classifier.py`)** — 정답 라벨이 수백 개 필요하지만 실제 광도곡선을 그만큼 받는 건 비현실적이라, Step 4에서 관찰한 패턴(행성일수록 신호·깊이가 큼)을 흉내 낸 **시뮬레이션 라벨**로 ML 파이프라인 전체(학습·검증·예측·시각화)를 익혔습니다. (테스트 정확도 ≈ 95%)
2. **실데이터 검증 (`05b_train_classifier_toi.py`)** — **NASA Exoplanet Archive의 TOI 카탈로그**(공개 오픈 데이터)에서 과학자들이 판정한 **진짜 라벨**을 받아, `CP`/`KP`(확정 행성) vs `FP`/`FA`(가짜신호)를 분류했습니다. 약 2,600개(행성 ≈ 1,400 / 가짜신호 ≈ 1,270)로 학습했고 **테스트 정확도 ≈ 80%**.
   - 가장 중요한 특징은 **행성 반지름**이었습니다. 가짜신호(대부분 쌍성)는 행성으로 가정하면 반지름이 비정상적으로 크게 계산되는데, 모델이 이 패턴을 스스로 학습해 실제 천문학자의 판별법을 재현했습니다.

> 💡 시뮬레이션의 95%보다 실데이터의 80%가 **더 의미 있는** 점수입니다. 실제 천체 데이터는 지저분하고 애매한 경우가 많으며, 이 격차 자체가 "깨끗한 가짜 데이터 vs 현실 데이터"의 차이를 보여줍니다.

---

## ❓ 막혔을 때

- `pip install`이 `lightkurve`에서 에러 → Python 3.12 가상환경이 맞는지 확인 (`python --version`)
- 그래프 창이 안 뜸 → `outputs\` 폴더의 `.png` 파일을 직접 열어보세요
- 다운로드가 느림/실패 → 잠시 후 재시도 (NASA 서버 혼잡일 수 있음)

## ▶️ 실행 순서 요약

환경 세팅(1~3) 후, 아래를 순서대로 실행하면 전체 파이프라인이 재현됩니다.

```powershell
python src\01_download_lightcurve.py   # Step 1: 데이터 받아 그리기
python src\02_flatten_lightcurve.py    # Step 2: flatten 전처리
python src\03_find_period_bls.py       # Step 3: BLS 주기 탐색 → Transit 확인
python src\04_extract_features.py      # Step 4: 별 여러 개 특징 추출 → features.csv
python src\05_train_classifier.py      # Step 5: 머신러닝 분류(시뮬레이션) → predictions.csv
python src\05b_train_classifier_toi.py # Step 5+: 진짜 라벨(NASA TOI)로 분류 → toi_labeled.csv
python src\06_api.py                   # Step 6+7: 서버 실행 후 http://127.0.0.1:8000/ 접속
```

> **🎉 프로젝트 완성!** 더 해보고 싶다면: `04_extract_features.py`의 `TARGETS` 목록에 원하는 별을 추가하거나, `web/index.html` 디자인을 바꿔 포트폴리오를 꾸며보세요.
