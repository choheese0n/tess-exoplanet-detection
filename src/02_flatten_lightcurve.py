"""
Step 2 — Light Curve 전처리 (flatten / 트렌드 제거)
---------------------------------------------------
Step 1에서 받은 별 'Pi Mensae'의 밝기 데이터는 전체적으로 울렁울렁
휘어 있습니다(= 트렌드). 이 트렌드는 행성과 상관없는 별/장비의 변화예요.

flatten() 으로 이 큰 물결을 펴서 평평하게 만들면,
행성이 지나갈 때 '아래로 쏙 떨어지는 Transit'만 또렷하게 남습니다.

이 스크립트는 '원본' vs '평평하게 편 것'을 위아래로 나란히 그려 비교합니다.

실행:
    python src\\02_flatten_lightcurve.py
"""

from pathlib import Path
import lightkurve as lk
import matplotlib.pyplot as plt

# 결과 저장 폴더 준비
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "Pi Mensae"   # Step 1과 같은 별

def main() -> None:
    print(f"[1/4] '{TARGET}' 의 TESS 관측 데이터를 검색합니다...")
    search = lk.search_lightcurve(TARGET, mission="TESS", author="SPOC")

    if len(search) == 0:
        print("데이터를 찾지 못했습니다. 별 이름/네트워크를 확인하세요.")
        return

    print("[2/4] 데이터를 다운로드합니다 (캐시가 있으면 빨라요)...")
    lc = search[0].download()

    # 결측치 제거 + 밝기 정규화(평균=1 기준) — Step 1과 동일한 기본 전처리
    print("[3/4] 기본 전처리(결측치 제거, 정규화)를 합니다...")
    lc = lc.remove_nans().normalize()

    # 핵심! flatten: 큰 트렌드(울렁임)를 펴서 평평하게 만듭니다.
    #   window_length = 트렌드를 추정할 '창'의 크기(점 개수, 홀수).
    #   값이 작을수록 더 촘촘하게 펴고, 클수록 완만하게 폅니다.
    print("[4/4] flatten() 으로 트렌드를 제거합니다...")
    flat_lc = lc.flatten(window_length=401)

    # 원본(위) vs 평평하게 편 것(아래)을 나란히 그리기
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    lc.plot(ax=ax1)
    ax1.set_title(f"Raw Light Curve - {TARGET}")

    flat_lc.plot(ax=ax2, color="tab:orange")
    ax2.set_title("After flatten (trend removed) - transits are easier to see")

    fig.tight_layout()

    out_path = OUTPUT_DIR / "pi_men_flatten.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n완료! 비교 그래프를 저장했습니다 → {out_path}")
    print("아래 그래프(주황색)가 더 평평하죠? 여기서 아래로 쏙 떨어지는 점이 Transit 후보입니다.")

    plt.show()


if __name__ == "__main__":
    main()
