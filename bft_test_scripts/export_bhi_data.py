#!/usr/bin/env python3
"""
BFT-BHI InfluxDB 데이터를 CSV로 내보내는 스크립트
SRS1 BFT-BHI 관련 데이터 추출용
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 한글 폰트 설정 (macOS: AppleGothic, Linux: NanumGothic)
import platform

if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = "AppleGothic"
else:  # Linux (EC2)
    plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# 프로젝트 루트 디렉토리를 Python 경로에 추가
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influxdb import InfluxDBClient


def plot_bhi_trend(df, output_file="bhi_trend.png"):
    """BFT_BHI_VALUE 트렌드 시각화

    Args:
        df: BHI 데이터프레임 (_time_gateway, BFT_BHI_VALUE 포함)
        output_file: 출력 이미지 파일명
    """
    if df.empty or "BFT_BHI_VALUE" not in df.columns:
        print("⚠️ 시각화할 BHI 데이터가 없습니다.")
        return None

    # 유효값만 필터링
    df_valid = df[df["BFT_BHI_VALUE"].notna()].copy()

    if len(df_valid) == 0:
        print("⚠️ 유효한 BHI 값이 없어 시각화를 건너뜁니다.")
        return None

    # _time_gateway를 datetime으로 변환
    if "_time_gateway" in df_valid.columns:
        df_valid["_time_gateway"] = pd.to_datetime(df_valid["_time_gateway"], utc=True)
        df_valid = df_valid.sort_values("_time_gateway")
    else:
        print("⚠️ _time_gateway 컬럼이 없습니다.")
        return None

    # 그래프 생성
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # BHI 트렌드 라인
    ax.plot(
        df_valid["_time_gateway"],
        df_valid["BFT_BHI_VALUE"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="steelblue",
        label="BHI (BFT Health Index)",
    )

    # 각 데이터 포인트 위에 BHI 값 표시 (소수점 2자리)
    for idx, row in df_valid.iterrows():
        ax.text(
            row["_time_gateway"],
            row["BFT_BHI_VALUE"] + 1.5,  # 포인트 위쪽에 표시
            f'{row["BFT_BHI_VALUE"]:.2f}',
            ha="center",
            va="bottom",
            fontsize=9,
            color="darkblue",
            fontweight="bold",
        )

    # 그래프 설정 (영어 라벨 사용)
    ax.set_xlabel("Date (UTC)", fontsize=12)
    ax.set_ylabel("BHI Value (%)", fontsize=12)

    date_range = f"{df_valid['_time_gateway'].min().strftime('%Y-%m-%d')} ~ {df_valid['_time_gateway'].max().strftime('%Y-%m-%d')}"
    ax.set_title(
        f"SRS1 Bag Filter Health Index (BHI) Trend\n"
        f"Period (UTC): {date_range} ({len(df_valid)} days)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    # Y축 범위 설정
    y_min = max(0, df_valid["BFT_BHI_VALUE"].min() - 10)
    y_max = min(100, df_valid["BFT_BHI_VALUE"].max() + 10)
    ax.set_ylim(y_min, y_max)

    # X축 날짜 포맷
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df_valid) // 15)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # 이미지 저장
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 트렌드 그래프 저장: {output_path.absolute()}")
    print(f"   • 파일: {output_path.name}")
    print(f"   • 데이터 포인트: {len(df_valid)}개")
    print(
        f"   • BHI 범위: {df_valid['BFT_BHI_VALUE'].min():.2f} ~ {df_valid['BFT_BHI_VALUE'].max():.2f}"
    )

    return output_path


def export_bhi_data(
    hours=360,  # 기본 15일 (BHI는 하루 1번 계산 -> 15개 유효값)
    output_file=None,
    measurement="SRS1",
    host="10.238.24.150",  # 개발기
    port=8086,
    username="read_user",
    password="!Skepinfluxuser25",
    database="SRS1",
):
    """
    BFT-BHI 데이터를 InfluxDB에서 CSV로 내보내기

    Args:
        hours: 조회할 시간 범위 (시간 단위, 기본 360시간=15일)
        output_file: 출력 파일명 (None이면 자동 생성)
        measurement: 측정값명
        host, port, username, password, database: InfluxDB 연결 정보

    Note:
        - BFT_BHI_VALUE는 하루 1번 계산됨
        - NULL 행은 자동 필터링하여 유효값만 저장
        - 15일 조회 시 약 15개의 유효 BHI 값 추출
    """

    # BFT-BHI 관련 컬럼
    bhi_columns = [
        "time",
        "_time_gateway",
        "BFT_BHI_CODE",
        "BFT_BHI_DATAQUALITY",
        "BFT_BHI_DATASTATUS",
        "BFT_BHI_VALUE",
    ]

    print("🔍 BFT-BHI InfluxDB 데이터 내보내기 시작")
    print("=" * 60)

    try:
        # InfluxDB 연결
        client = InfluxDBClient(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        )

        print(f"🔗 InfluxDB 연결: {host}:{port}/{database}")
        print(f"📊 Measurement: {measurement}")

        # 시간 범위 설정
        now = datetime.utcnow()
        start_time = now - timedelta(hours=hours)

        start_utc = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        print(
            f"📅 조회 기간: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        print(f"📊 조회 범위: {hours}시간 ({hours/24:.1f}일)")

        # 쿼리 생성 (BFT_BHI_VALUE가 유효한 행만 조회)
        # 분할 쿼리: 하루씩 조회하여 메모리 부족 방지
        print(f"📌 메모리 효율을 위해 하루씩 분할 조회합니다...")
        columns_str = ", ".join([f'"{col}"' for col in bhi_columns])

        # 하루씩 분할 조회
        all_dfs = []
        num_days = int(hours / 24) + 1

        for day in range(num_days):
            day_start = start_time + timedelta(days=day)
            day_end = min(day_start + timedelta(days=1), now)

            day_start_str = day_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            day_end_str = day_end.strftime("%Y-%m-%dT%H:%M:%SZ")

            query = f"""
            SELECT {columns_str} FROM "{measurement}" 
            WHERE time >= '{day_start_str}' AND time < '{day_end_str}'
            ORDER BY time ASC
            """

            print(
                f"   Day {day+1}/{num_days}: {day_start.strftime('%Y-%m-%d')}...",
                end=" ",
            )

            try:
                result = client.query(query)
                day_df = pd.DataFrame(list(result.get_points()))

                if not day_df.empty:
                    all_dfs.append(day_df)
                    # BHI 값이 있는지 확인
                    if "BFT_BHI_VALUE" in day_df.columns:
                        valid_count = day_df["BFT_BHI_VALUE"].notna().sum()
                        print(f"✅ {len(day_df):,}행 (BHI 유효값: {valid_count}개)")
                    else:
                        print(f"✅ {len(day_df):,}행")
                else:
                    print("⚠️ 데이터 없음")
            except Exception as e:
                print(f"❌ 오류: {e}")
                continue

        # 모든 데이터 합치기
        if not all_dfs:
            print("\n❌ 조회된 데이터가 없습니다.")
            return None

        df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n✅ 전체 조회 완료: {len(df):,} 행, {len(df.columns)} 컬럼")

        # 시간 컬럼 변환
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            print(f"📅 데이터 시간 범위: {df['time'].min()} ~ {df['time'].max()}")

        # BFT_BHI_VALUE 통계 (필터링 전)
        if "BFT_BHI_VALUE" in df.columns:
            print(f"\n📈 BFT_BHI_VALUE 통계:")
            total_rows = len(df)
            null_count = df["BFT_BHI_VALUE"].isnull().sum()
            valid_count = df["BFT_BHI_VALUE"].notna().sum()

            print(f"   - 총 행 수: {total_rows:,}")
            print(f"   - NULL 값: {null_count:,} ({null_count/total_rows*100:.2f}%)")
            print(f"   - 유효값: {valid_count:,} ({valid_count/total_rows*100:.2f}%)")

            if valid_count > 0:
                print(
                    f"   - 유효값 범위: {df['BFT_BHI_VALUE'].min():.2f} ~ {df['BFT_BHI_VALUE'].max():.2f}"
                )
                print(f"   - 평균: {df['BFT_BHI_VALUE'].mean():.2f}")

        # BFT_BHI_VALUE가 NULL인 행 필터링 (유효값만 남김)
        if "BFT_BHI_VALUE" in df.columns:
            df_filtered = df[df["BFT_BHI_VALUE"].notna()].copy()

            if len(df_filtered) == 0:
                print("\n⚠️ BFT_BHI_VALUE 유효값이 없습니다. 전체 데이터를 저장합니다.")
                df_filtered = df
            else:
                print(f"\n🔍 NULL 행 필터링:")
                print(f"   - 필터링 전: {len(df):,} 행")
                print(f"   - 필터링 후: {len(df_filtered):,} 행 (유효값만)")
                print(f"   - 제거된 행: {len(df) - len(df_filtered):,} 행")
                df = df_filtered

        # 출력 파일명 생성
        if output_file is None:
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            output_file = f"bhi_data_{timestamp}_{hours}h.csv"

        # CSV 저장
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)

        print(f"\n💾 CSV 저장 완료: {output_path.absolute()}")
        print(f"📊 파일 크기: {output_path.stat().st_size / 1024:.1f} KB")

        # 데이터 미리보기
        print("\n📋 데이터 미리보기 (처음 5행):")
        print(df.head())

        print(f"\n✅ 내보내기 완료!")
        print(f"   • 파일: {output_path.name}")
        print(f"   • 경로: {output_path.absolute()}")
        print(f"   • 행 수: {len(df):,}")
        print(f"   • 컬럼 수: {len(df.columns)}")

        # BHI 트렌드 시각화
        print(f"\n📊 BHI 트렌드 시각화 중...")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        plot_file = f"bhi_trend_{timestamp}_{hours}h.png"
        plot_path = plot_bhi_trend(df, output_file=plot_file)

        # GitHub을 통한 데이터 전송 안내
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n🔄 GitHub을 통한 데이터 전송 방법:")

        if file_size_mb > 25:
            print(
                f"   ⚠️ CSV 파일이 {file_size_mb:.1f}MB로 GitHub 제한(25MB)을 초과합니다."
            )
            print(f"   💡 --hours 옵션으로 데이터 범위를 줄이세요.")
        else:
            files_to_add = [output_path.name]
            if plot_path:
                files_to_add.append(plot_path.name)

            print(f"   1. git add {' '.join(files_to_add)}")
            print(f"   2. git commit -m 'Add BHI data and trend: {output_path.name}'")
            print(f"   3. git push origin main")
            print(f"   4. 로컬에서 git pull origin main 후 파일 사용")

        return output_path

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="BFT-BHI InfluxDB 데이터를 CSV로 내보내기"
    )

    parser.add_argument(
        "--hours",
        type=float,
        default=360,
        help="조회할 시간 범위 (기본: 360시간=15일, BHI는 하루 1번 계산)",
    )
    parser.add_argument("--output", "-o", help="출력 파일명 (기본: 자동 생성)")
    parser.add_argument(
        "--measurement", "-m", default="SRS1", help="측정값명 (기본: SRS1)"
    )

    args = parser.parse_args()

    # BHI 데이터 내보내기 실행
    result = export_bhi_data(
        hours=args.hours,
        output_file=args.output,
        measurement=args.measurement,
    )

    if result:
        print(f"\n🎉 BFT-BHI 데이터 내보내기 완료!")
    else:
        print(f"\n❌ BFT-BHI 데이터 내보내기 실패")


if __name__ == "__main__":
    main()
