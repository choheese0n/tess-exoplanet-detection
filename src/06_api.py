"""
Step 6 — 결과를 웹 API로 제공하기 (FastAPI)
--------------------------------------------
지금까지 만든 '외계행성 분류'를 웹 주소(API)로 열어, 브라우저나 다른
프로그램이 불러갈 수 있게 합니다. (주방에서 요리만 하다가 '주문 창구'를 다는 셈)

서버를 켜면 아래 주소들이 열립니다:
  http://127.0.0.1:8000/            → 안내
  http://127.0.0.1:8000/stars       → 진짜 별 5개 예측 결과 (JSON)
  http://127.0.0.1:8000/stars/WASP-18 → 특정 별 하나
  http://127.0.0.1:8000/predict     → 숫자를 넣으면 행성 확률 계산 (POST)
  http://127.0.0.1:8000/docs        → ★ 브라우저에서 눌러보는 자동 테스트 화면 ★

실행:
    python src\\06_api.py
  그 뒤 브라우저에서 http://127.0.0.1:8000/docs 를 열어보세요.
  끄려면 터미널에서 Ctrl + C.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import uvicorn

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FEATURES = ["depth", "duration", "log_power"]

# ---------------------------------------------------------------
# 서버가 켜질 때 '모델'을 한 번 학습해 둡니다 (Step 5와 같은 방식, 다운로드 없음)
# ---------------------------------------------------------------
def build_model() -> RandomForestClassifier:
    rng = np.random.default_rng(42)
    n = 200
    # 행성 없음(0): 신호 약함 / 행성 있음(1): 신호 셈 (현실처럼 일부 겹침)
    power0 = rng.lognormal(np.log(40), 0.5, n);   depth0 = np.abs(rng.normal(0, 0.0003, n))
    power1 = rng.lognormal(np.log(600), 1.1, n);  depth1 = rng.lognormal(np.log(0.0012), 0.8, n)
    dur0 = rng.uniform(0.03, 0.15, n);            dur1 = rng.uniform(0.03, 0.15, n)

    df = pd.DataFrame({
        "depth": np.concatenate([depth0, depth1]),
        "duration": np.concatenate([dur0, dur1]),
        "bls_power": np.concatenate([power0, power1]),
        "label": np.concatenate([np.zeros(n), np.ones(n)]).astype(int),
    })
    df["log_power"] = np.log10(df["bls_power"])

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(df[FEATURES].values, df["label"].values)
    return clf


MODEL = build_model()

app = FastAPI(
    title="TESS 외계행성 탐지 API",
    description="별의 특징(깊이·신호세기 등)을 받아 '외계행성일 확률'을 돌려줍니다.",
    version="1.0.0",
)


def predict_proba(depth: float, duration: float, bls_power: float) -> float:
    """특징 3개를 받아 '행성일 확률(0~1)'을 계산."""
    log_power = float(np.log10(bls_power))
    X = np.array([[depth, duration, log_power]])
    return float(MODEL.predict_proba(X)[0, 1])


# ---- 입력 형식 정의 (/predict 용) ----
class StarFeatures(BaseModel):
    depth: float        # Transit 깊이 (예: 0.0002)
    duration: float     # Transit 지속시간 (일, 예: 0.1)
    bls_power: float    # BLS 신호 세기 (예: 500)


# ---- 엔드포인트들 ----
@app.get("/")
def root():
    return {
        "message": "TESS 외계행성 탐지 API에 오신 걸 환영합니다 🪐",
        "try": ["/stars", "/stars/WASP-18", "/predict (POST)", "/docs (테스트 화면)"],
    }


@app.get("/stars")
def list_stars():
    """Step 4에서 뽑은 진짜 별 5개 + 각 별의 행성 확률."""
    feat_path = OUTPUT_DIR / "features.csv"
    if not feat_path.exists():
        raise HTTPException(status_code=404,
                            detail="features.csv 없음. 먼저 04_extract_features.py 실행")
    df = pd.read_csv(feat_path)
    results = []
    for _, r in df.iterrows():
        p = predict_proba(r["depth"], r["duration"], r["bls_power"])
        results.append({
            "target": r["target"],
            "best_period": round(float(r["best_period"]), 4),
            "depth": float(r["depth"]),
            "bls_power": float(r["bls_power"]),
            "planet_probability": round(p, 3),
            "verdict": "행성 후보" if p >= 0.5 else "아님",
        })
    return {"count": len(results), "stars": results}


@app.get("/stars/{name}")
def get_star(name: str):
    """별 이름 하나로 조회 (대소문자 무시)."""
    data = list_stars()
    for s in data["stars"]:
        if s["target"].lower() == name.lower():
            return s
    raise HTTPException(status_code=404, detail=f"'{name}' 별을 찾지 못했습니다")


@app.post("/predict")
def predict(features: StarFeatures):
    """숫자 특징을 직접 넣어 행성 확률을 계산."""
    p = predict_proba(features.depth, features.duration, features.bls_power)
    return {
        "input": features.model_dump(),
        "planet_probability": round(p, 3),
        "verdict": "행성 후보" if p >= 0.5 else "아님",
    }


if __name__ == "__main__":
    print("서버를 시작합니다. 브라우저에서 http://127.0.0.1:8000/docs 를 열어보세요.")
    print("(끄려면 이 창에서 Ctrl + C)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
