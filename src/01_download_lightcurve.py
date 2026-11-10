"""
Step 1 — TESS Light Curve 다운로드 & 시각화
--------------------------------------------
NASA TESS가 관측한 별 'Pi Mensae'(외계행성 Pi Men c를 가진 별)의
밝기 시계열 데이터를 받아 그래프로 그립니다.

밝기가 '아래로 뾰족하게' 떨어지는 지점 = 행성이 별 앞을 지나간 Transit 입니다.

실행:
    python src\\01_download_lightcurve.py
"""

from pathlib import Path
import lightkurve as lk
import matplotlib.pyplot as plt

# 결과 저장 폴더 준비
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Pi Mensae"   # 분석할 별 (원하면 "TIC 261136679" 같은 ID로 바꿔도 됨)

def main() -> None:
    print(f"[1/4] '{TARGET}' 의 TESS 관측 데이터를 검색합니다...")
    search = lk.search_lightcurve(TARGET, mission="TESS", author="SPOC")

    if len(search) == 0:
        print("데이터를 찾지 못했습니다. 별 이름/네트워크를 확인하세요.")
        return

    print(f"    총 {len(search)}개의 관측(sector)이 있습니다. 첫 번째 것을 사용합니다.")
    print("[2/4] 데이터를 다운로드합니다 (처음엔 1~2분 걸릴 수 있어요)...")
    lc = search[0].download()

    # 결측치 제거 + 밝기 정규화(평균=1 기준)
    print("[3/4] 간단 전처리(결측치 제거, 정규화)를 합니다...")
    lc = lc.remove_nans().normalize()

    # 그래프 그리기
    print("[4/4] 그래프를 그립니다...")
    ax = lc.plot()
    ax.set_title(f"TESS Light Curve - {TARGET}")

    out_path = OUTPUT_DIR / "pi_men_lightcurve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n완료! 그래프를 저장했습니다 → {out_path}")
    print("아래로 뾰족하게 떨어지는 지점이 바로 외계행성 Transit 입니다.")

    plt.show()


if __name__ == "__main__":
    main()
