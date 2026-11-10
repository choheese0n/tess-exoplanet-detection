"""
Step 5 — 머신러닝으로 '행성 후보 vs 아님' 분류하기 (scikit-learn)
------------------------------------------------------------------
[정직한 안내]
머신러닝 학습에는 '정답(label)이 붙은 데이터 수백 개'가 필요한데,
진짜 TESS 별을 그만큼 받으려면 다운로드에 몇 시간이 걸립니다(초보 실습엔 비현실적).

그래서 이 스크립트는:
  (1) 훈련용 데이터를 '시뮬레이션'으로 수백 개 만듭니다.
      - Step 4에서 본 진짜 패턴을 흉내: 행성 있는 별은 depth/bls_power가 큰 편,
        없는 별은 작은 편. (현실처럼 애매하게 겹치는 구간도 일부러 포함)
      - 이는 ML '파이프라인'을 배우기 위한 교보재이며, 진짜 NASA 라벨이 아닙니다.
  (2) 이 데이터로 Random Forest 분류기를 학습하고 정확도를 확인하고,
  (3) 학습된 모델을 Step 4의 진짜 별 5개(outputs/features.csv)에 적용해
      '행성일 확률'을 예측합니다. ← 진짜 데이터로 마무리!

결과:
  - 콘솔: 정확도, 혼동행렬, 진짜 별 5개의 예측 확률
  - outputs/ml_feature_space.png : 특징 공간 그림(행성/비행성 분포 + 진짜 별 위치)

실행:
    python src\\05_train_classifier.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)   # 재현성(실행할 때마다 같은 결과)

# 모델이 사용할 특징들
FEATURES = ["depth", "duration", "log_power"]


def make_synthetic(n_planet: int = 200, n_noise: int = 200) -> pd.DataFrame:
    """Step 4의 실제 패턴을 흉내 낸 '라벨 붙은' 훈련 데이터(가짜)를 만든다."""
    # --- 행성 없음 (label=0): 신호 약하고 깊이 얕음 ---
    power0 = rng.lognormal(mean=np.log(40), sigma=0.5, size=n_noise)
    depth0 = np.abs(rng.normal(0, 0.0003, size=n_noise))
    dur0 = rng.uniform(0.03, 0.15, size=n_noise)

    # --- 행성 있음 (label=1): 신호 세고 깊이 깊은 편 (단, 일부는 약하게 겹침) ---
    power1 = rng.lognormal(mean=np.log(600), sigma=1.1, size=n_planet)
    depth1 = rng.lognormal(mean=np.log(0.0012), sigma=0.8, size=n_planet)
    dur1 = rng.uniform(0.03, 0.15, size=n_planet)

    df = pd.DataFrame({
        "depth":     np.concatenate([depth0, depth1]),
        "duration":  np.concatenate([dur0, dur1]),
        "bls_power": np.concatenate([power0, power1]),
        "label":     np.concatenate([np.zeros(n_noise), np.ones(n_planet)]).astype(int),
    })
    df["log_power"] = np.log10(df["bls_power"])   # 신호 세기는 범위가 넓어 log로 변환
    return df


def main() -> None:
    print("[1/4] 훈련용 시뮬레이션 데이터를 생성합니다 (행성 200 + 비행성 200)...")
    data = make_synthetic()

    X = data[FEATURES].values
    y = data["label"].values

    # 데이터를 '학습용 70% / 시험용 30%'로 나눔 (시험용은 채점에만 사용)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("[2/4] Random Forest 분류기를 학습시킵니다...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    print("[3/4] 시험용 데이터로 채점합니다...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n   정확도(accuracy): {acc*100:.1f}%")
    print("   혼동행렬 [행=실제, 열=예측]  (좌상·우하가 '맞춘 것'):")
    print("   " + str(confusion_matrix(y_test, y_pred)).replace("\n", "\n   "))
    print("\n   상세 리포트:")
    print("   " + classification_report(y_test, y_pred,
                                        target_names=["비행성(0)", "행성(1)"]
                                        ).replace("\n", "\n   "))

    # 어떤 특징이 판단에 중요했는지
    importances = dict(zip(FEATURES, clf.feature_importances_))
    print("   특징 중요도:", {k: round(v, 3) for k, v in importances.items()})

    # --- (4) 학습된 모델을 '진짜 별 5개'에 적용 ---
    print("\n[4/4] 학습된 모델을 Step 4의 진짜 별들에 적용합니다...")
    feat_path = OUTPUT_DIR / "features.csv"
    if not feat_path.exists():
        print(f"   {feat_path} 가 없습니다. 먼저 04_extract_features.py 를 실행하세요.")
        return

    real = pd.read_csv(feat_path)
    real["log_power"] = np.log10(real["bls_power"])
    proba = clf.predict_proba(real[FEATURES].values)[:, 1]   # 행성(1)일 확률
    real["planet_probability"] = proba
    real["예측"] = np.where(proba >= 0.5, "행성 후보 ✔", "아님")

    show = real[["target", "best_period", "depth", "bls_power",
                 "planet_probability", "예측"]].copy()
    show["planet_probability"] = (show["planet_probability"] * 100).round(1)
    print("\n   === 진짜 별 5개 예측 결과 ===")
    print("   " + show.to_string(index=False).replace("\n", "\n   "))

    real.to_csv(OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")

    # --- 그림: 특징 공간(행성/비행성 분포) + 진짜 별 위치 ---
    plt.figure(figsize=(9, 6))
    m0 = data["label"] == 0
    m1 = data["label"] == 1
    plt.scatter(data.loc[m0, "log_power"], data.loc[m0, "depth"],
                s=12, alpha=0.3, color="tab:gray", label="train: no planet (sim)")
    plt.scatter(data.loc[m1, "log_power"], data.loc[m1, "depth"],
                s=12, alpha=0.3, color="tab:blue", label="train: planet (sim)")
    plt.scatter(real["log_power"], real["depth"],
                s=180, marker="*", color="red", edgecolor="black",
                zorder=5, label="REAL TESS stars")
    for _, r in real.iterrows():
        plt.annotate(r["target"], (r["log_power"], r["depth"]),
                     fontsize=9, xytext=(5, 5), textcoords="offset points")
    plt.xlabel("log10(BLS power)  →  signal strength")
    plt.ylabel("transit depth")
    plt.yscale("log")
    plt.title("Feature space: planets vs non-planets (red stars = real TESS targets)")
    plt.legend()
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "ml_feature_space.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n완료!")
    print(f"  - 예측 결과표 → {OUTPUT_DIR / 'predictions.csv'}")
    print(f"  - 특징 공간 그림 → {fig_path}")


if __name__ == "__main__":
    main()
