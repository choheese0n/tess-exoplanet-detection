"""
Step 5 (진짜 라벨 버전) — NASA TOI 카탈로그로 '확정행성 vs 가짜신호' 분류
--------------------------------------------------------------------------
앞의 05_train_classifier.py 는 '가짜(시뮬레이션) 라벨'로 ML 파이프라인을
연습한 버전이었습니다. 이 스크립트는 그걸 '진짜 라벨'로 업그레이드합니다.

데이터: NASA Exoplanet Archive의 TOI(TESS Objects of Interest) 카탈로그
        (공개 오픈 데이터, TAP 서비스로 자동 다운로드)

라벨(tfopwg_disp 기준):
    CP, KP (확정/기존 행성)        → 1 (진짜 행성)
    FP, FA (가짜신호/오경보)       → 0 (행성 아님)
    PC, APC (아직 미확정)          → 제외 (과학자도 확신 못 한 것)

특징(카탈로그 측정값):
    pl_orbper   공전주기(일)
    pl_trandurh Transit 지속시간(시간)
    pl_trandep  Transit 깊이(ppm)
    pl_rade     행성 반지름(지구=1)   ← 가짜신호는 비정상적으로 큰 경우 많음
    st_tmag     별 밝기(TESS 등급)

결과:
    outputs/toi_labeled.csv         (정제된 '라벨 붙은' 진짜 데이터)
    outputs/toi_feature_importance.png
    콘솔: 데이터 개수, 정확도, 혼동행렬, 특징 중요도

실행:
    python src\\05b_train_classifier_toi.py
"""

from pathlib import Path
from urllib.parse import quote
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = ["pl_orbper", "pl_trandurh", "pl_trandep", "pl_rade", "st_tmag"]
FEATURE_KO = {
    "pl_orbper": "공전주기(일)",
    "pl_trandurh": "지속시간(시간)",
    "pl_trandep": "깊이(ppm)",
    "pl_rade": "행성반지름(지구=1)",
    "st_tmag": "별밝기(TESS등급)",
}
# 그래프용 영어 라벨 (matplotlib 한글 폰트 깨짐 방지)
FEATURE_EN = {
    "pl_orbper": "orbital period",
    "pl_trandurh": "transit duration",
    "pl_trandep": "transit depth (ppm)",
    "pl_rade": "planet radius",
    "st_tmag": "star brightness",
}

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def download_toi() -> pd.DataFrame:
    """NASA Exoplanet Archive에서 TOI 카탈로그를 내려받는다."""
    cols = "toi,tid,tfopwg_disp," + ",".join(FEATURES)
    adql = f"select {cols} from toi"
    url = f"{TAP}?query={quote(adql)}&format=csv"
    print("  · NASA Exoplanet Archive에서 TOI 카탈로그 다운로드 중...")
    df = pd.read_csv(url)
    print(f"    받은 전체 TOI 행: {len(df)}개")
    return df


def main() -> None:
    print("[1/4] 진짜 라벨 데이터(TOI)를 가져옵니다...")
    try:
        df = download_toi()
    except Exception as e:
        print(f"    다운로드 실패 ({type(e).__name__}: {e}). 네트워크를 확인하세요.")
        return

    print("[2/4] 라벨을 붙이고 데이터를 정제합니다...")
    # 라벨 매핑: 확정행성=1, 가짜신호=0, 그 외(미확정)=제외
    label_map = {"CP": 1, "KP": 1, "FP": 0, "FA": 0}
    df["label"] = df["tfopwg_disp"].map(label_map)

    # 라벨이 있고(확정/가짜), 특징에 빈 값이 없는 행만 사용
    data = df.dropna(subset=["label"] + FEATURES).copy()
    data["label"] = data["label"].astype(int)

    n_planet = int((data["label"] == 1).sum())
    n_fp = int((data["label"] == 0).sum())
    print(f"    사용할 데이터: 총 {len(data)}개  (진짜 행성 {n_planet} / 가짜신호 {n_fp})")

    if len(data) < 50 or n_planet < 10 or n_fp < 10:
        print("    데이터가 너무 적습니다. 카탈로그 응답을 확인하세요.")
        return

    X = data[FEATURES].values
    y = data["label"].values

    # 학습 70% / 시험 30% (클래스 비율 유지)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("[3/4] Random Forest 분류기를 학습시킵니다 (진짜 라벨로!)...")
    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    print("[4/4] 시험용 데이터로 채점합니다...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n   === 결과 (진짜 TOI 라벨) ===")
    print(f"   정확도(accuracy): {acc*100:.1f}%")
    print("   혼동행렬 [행=실제, 열=예측]  (좌상·우하가 '맞춘 것'):")
    print("   " + str(confusion_matrix(y_test, y_pred)).replace("\n", "\n   "))
    print("\n   상세 리포트:")
    print("   " + classification_report(
        y_test, y_pred, target_names=["가짜신호(0)", "행성(1)"]
    ).replace("\n", "\n   "))

    importances = sorted(zip(FEATURES, clf.feature_importances_),
                         key=lambda kv: kv[1], reverse=True)
    print("   특징 중요도(높을수록 판단에 중요):")
    for f, imp in importances:
        print(f"     - {FEATURE_KO[f]:<18} {imp:.3f}")

    # 정제된 라벨 데이터 저장
    out_csv = OUTPUT_DIR / "toi_labeled.csv"
    data[["toi", "tid", "tfopwg_disp", "label"] + FEATURES].to_csv(
        out_csv, index=False, encoding="utf-8-sig")

    # 특징 중요도 그래프
    feats = [FEATURE_EN[f] for f, _ in importances][::-1]
    vals = [imp for _, imp in importances][::-1]
    plt.figure(figsize=(8, 4.5))
    plt.barh(range(len(feats)), vals, color="#2a78d6")
    plt.yticks(range(len(feats)), feats)
    plt.xlabel("feature importance")
    plt.title(f"TOI real-label model - feature importance (accuracy {acc*100:.1f}%)")
    for i, v in enumerate(vals):
        plt.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    plt.tight_layout()
    fig_path = OUTPUT_DIR / "toi_feature_importance.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n완료!")
    print(f"  - 라벨 데이터 → {out_csv}")
    print(f"  - 특징 중요도 그림 → {fig_path}")
    print("이번엔 '진짜 NASA 라벨'로 학습했으니, 과학적으로도 의미 있는 분류기입니다.")


if __name__ == "__main__":
    main()
