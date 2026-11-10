"""
Step 4 — 여러 별에서 '특징(feature)'을 뽑아 표(CSV)로 정리하기
-------------------------------------------------------------
머신러닝(Step 5)은 그래프 그림이 아니라 '숫자 표'를 학습합니다.
그래서 별마다 Step 1~3 과정(다운로드 → 전처리 → BLS)을 자동으로 돌려
아래 숫자 특징들을 뽑아 한 줄씩 표로 쌓습니다.

  - best_period : BLS가 찾은 주기 (일)
  - depth       : Transit 깊이 (별빛이 가려지는 정도)
  - duration    : Transit 지속 시간 (일)
  - bls_power   : 신호 세기 (클수록 진짜 행성일 가능성 ↑)
  - n_points    : 데이터 점 개수

결과: outputs/features.csv  (별 = 행, 특징 = 열)

실행:
    python src\\04_extract_features.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import lightkurve as lk

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 분석할 별 목록 (모두 TESS가 관측한 유명한 외계행성 보유 별들)
TARGETS = [
    "Pi Mensae",
    "TOI-700",
    "LHS 3844",
    "WASP-18",
    "HD 21749",
]

def extract_one(target: str) -> dict | None:
    """별 하나를 처리해 특징 dict를 돌려준다. 실패하면 None."""
    print(f"  · '{target}' 처리 중...", end=" ", flush=True)
    search = lk.search_lightcurve(target, mission="TESS", author="SPOC")
    if len(search) == 0:
        print("데이터 없음 → 건너뜀")
        return None

    # 전처리: 결측치 제거 → 정규화 → flatten
    lc = search[0].download().remove_nans().normalize().flatten(window_length=401)

    # BLS로 주기 탐색 (1~15일)
    periods = np.linspace(1, 15, 8000)
    bls = lc.to_periodogram(method="bls", period=periods, frequency_factor=500)

    row = {
        "target": target,
        "best_period": float(bls.period_at_max_power.value),
        "depth": float(bls.depth_at_max_power),
        "duration": float(bls.duration_at_max_power.value),
        "bls_power": float(bls.max_power.value),
        "n_points": int(len(lc.flux)),
    }
    print(f"완료 (주기 {row['best_period']:.2f}일)")
    return row


def main() -> None:
    print(f"[1/2] 별 {len(TARGETS)}개에서 특징을 추출합니다 (다운로드로 몇 분 걸릴 수 있어요)...")
    rows = []
    for t in TARGETS:
        try:
            r = extract_one(t)
            if r is not None:
                rows.append(r)
        except Exception as e:
            print(f"에러로 건너뜀 ({type(e).__name__}: {e})")

    if not rows:
        print("추출된 데이터가 없습니다. 네트워크를 확인하세요.")
        return

    df = pd.DataFrame(rows)

    print("\n[2/2] 추출한 특징 표:")
    print(df.to_string(index=False))

    out_path = OUTPUT_DIR / "features.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n완료! 특징 표를 저장했습니다 → {out_path}")
    print("이 표가 Step 5 머신러닝의 '교재'가 됩니다.")


if __name__ == "__main__":
    main()
