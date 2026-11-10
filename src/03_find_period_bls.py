"""
Step 3 — BLS로 행성의 '주기'를 찾고 Transit을 눈에 보이게 만들기
----------------------------------------------------------------
행성 Transit은 너무 얕아 눈으로는 안 보입니다. 하지만 행성은 '규칙적으로'
별 앞을 지나갑니다(Pi Men c는 약 6일마다). 그래서:

  1) 온갖 주기를 다 시험하는 BLS(Box Least Squares) 알고리즘으로
     'Transit이 가장 선명해지는 주기'를 찾고,
  2) 그 주기로 데이터를 '접으면(fold)' 수십 번의 Transit이 겹쳐 쌓여
     노이즈가 상쇄되고 U자 모양 Transit이 또렷하게 드러납니다.

결과 그래프 2개를 저장합니다:
  - outputs/pi_men_bls_periodogram.png  (주기 스펙트럼: 뾰족한 봉우리 = 정답 주기)
  - outputs/pi_men_folded.png           (최적 주기로 접은 그래프: U자 Transit)

실행:
    python src\\03_find_period_bls.py
"""

from pathlib import Path
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Pi Mensae"

def main() -> None:
    print(f"[1/5] '{TARGET}' 의 TESS 데이터를 가져옵니다...")
    search = lk.search_lightcurve(TARGET, mission="TESS", author="SPOC")
    if len(search) == 0:
        print("데이터를 찾지 못했습니다. 별 이름/네트워크를 확인하세요.")
        return
    lc = search[0].download()

    print("[2/5] 전처리(결측치 제거, 정규화, flatten)를 합니다...")
    lc = lc.remove_nans().normalize().flatten(window_length=401)

    # BLS가 시험해 볼 주기 범위 (하루 ~ 15일 사이를 촘촘히 탐색)
    print("[3/5] BLS로 '어느 주기에서 Transit이 가장 선명한지' 탐색합니다...")
    periods = np.linspace(1, 15, 10000)   # 1~15일 사이 10000개 후보 주기
    bls = lc.to_periodogram(method="bls", period=periods, frequency_factor=500)

    # 신호가 가장 강한 지점 = 가장 그럴듯한 주기
    best_period = bls.period_at_max_power
    best_t0 = bls.transit_time_at_max_power
    best_dur = bls.duration_at_max_power
    print(f"    → 찾은 최적 주기: {best_period:.4f}")
    print(f"      (참고: Pi Men c의 실제 주기는 약 6.27일입니다)")

    # (1) 주기 스펙트럼 그리기
    print("[4/5] 주기 스펙트럼 그래프를 저장합니다...")
    ax = bls.plot()
    ax.set_title(f"BLS Periodogram - {TARGET} (peak = best period)")
    ax.axvline(best_period.value, color="red", ls="--", alpha=0.6, label="best period")
    ax.legend()
    p1 = OUTPUT_DIR / "pi_men_bls_periodogram.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()

    # (2) 최적 주기로 접어서(fold) U자 Transit 보기
    print("[5/5] 최적 주기로 접은(folded) 그래프를 저장합니다...")
    folded = lc.fold(period=best_period, epoch_time=best_t0)
    ax = folded.scatter(s=2, alpha=0.4)
    ax.set_title(f"Folded Light Curve (period = {best_period:.3f}) - dip = transit")
    ax.set_xlim(-0.5 * best_dur.value * 5, 0.5 * best_dur.value * 5)  # Transit 부근 확대
    p2 = OUTPUT_DIR / "pi_men_folded.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n완료!")
    print(f"  - 주기 스펙트럼 → {p1}")
    print(f"  - 접은 그래프   → {p2}")
    print("접은 그래프 가운데가 U자로 쏙 내려가 있으면, 그게 바로 외계행성 Transit입니다!")


if __name__ == "__main__":
    main()
