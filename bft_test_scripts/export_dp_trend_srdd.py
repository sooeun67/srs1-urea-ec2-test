#!/usr/bin/env python3
"""
SRDD 플랜트 차압(Differential Pressure) 트렌드 시각화 스크립트
목적: BHI 값 상승 원인 분석을 위한 차압 패턴 확인
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


def plot_dp_trend(df, output_file="dp_trend_srdd.png"):
    """차압 트렌드 시각화

    Args:
        df: 차압 데이터프레임 (_time_gateway, BFT_EQ_FG_DP_1 포함)
        output_file: 출력 이미지 파일명
    """
    if df.empty or "BFT_EQ_FG_DP_1" not in df.columns:
        print("⚠️ 시각화할 차압 데이터가 없습니다.")
        return None

    # 유효값만 필터링
    df_valid = df[df["BFT_EQ_FG_DP_1"].notna()].copy()

    if len(df_valid) == 0:
        print("⚠️ 유효한 차압 값이 없어 시각화를 건너뜁니다.")
        return None

    # _time_gateway를 datetime으로 변환
    if "_time_gateway" in df_valid.columns:
        df_valid["_time_gateway"] = pd.to_datetime(df_valid["_time_gateway"], utc=True)
        df_valid = df_valid.sort_values("_time_gateway")
    else:
        print("⚠️ _time_gateway 컬럼이 없습니다.")
        return None

    # 그래프 생성
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # 차압 트렌드 라인
    ax.plot(
        df_valid["_time_gateway"],
        df_valid["BFT_EQ_FG_DP_1"],
        linewidth=1.5,
        color="darkblue",
        alpha=0.8,
        label="Differential Pressure (BFT_EQ_FG_DP_1)",
    )

    # 그래프 설정 (영어 라벨 사용)
    ax.set_xlabel("Date (UTC)", fontsize=12)
    ax.set_ylabel("Average Differential Pressure (Pa)", fontsize=12)

    date_range = f"{df_valid['_time_gateway'].min().strftime('%Y-%m-%d')} ~ {df_valid['_time_gateway'].max().strftime('%Y-%m-%d')}"
    ax.set_title(
        f"SRDD Bag Filter Differential Pressure Trend\n"
        f"Period (UTC): {date_range} ({len(df_valid)} data points)\n"
        f"Purpose: Analyze pressure pattern for BHI increase cause",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    # Y축 범위 설정 (차압 특성에 맞게 조정)
    y_min = max(0, df_valid["BFT_EQ_FG_DP_1"].min() - 10)
    y_max = df_valid["BFT_EQ_FG_DP_1"].max() + 20
    ax.set_ylim(y_min, y_max)

    # TA 기간 배경색 추가 (10/07~10/11)
    if not df_valid.empty:
        # 날짜 범위 확인
        min_date = df_valid["_time_gateway"].min()
        max_date = df_valid["_time_gateway"].max()

        # 2025년 10월 7일~11일 UTC 기간
        ta_start = pd.Timestamp("2025-10-07 00:00:00", tz="UTC")
        ta_end = pd.Timestamp("2025-10-11 23:59:59", tz="UTC")

        # 그래프 날짜 범위와 TA 기간이 겹치는 경우에만 표시
        if ta_start <= max_date and ta_end >= min_date:
            # 실제 표시할 TA 기간 계산
            display_ta_start = max(ta_start, min_date)
            display_ta_end = min(ta_end, max_date)

            # 옅은 주황색 배경 추가
            ax.axvspan(
                display_ta_start,
                display_ta_end,
                alpha=0.2,
                color="orange",
                label="TA Period (10/07-10/11)",
            )

            # 범례에 TA 기간 추가
            ax.legend(loc="upper left", fontsize=10)

    # X축 날짜 포맷 (매일 표시)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # 통계 정보 텍스트 추가
    stats_text = f"""Statistics:
Min: {df_valid['BFT_EQ_FG_DP_1'].min():.1f} Pa
Max: {df_valid['BFT_EQ_FG_DP_1'].max():.1f} Pa
Mean: {df_valid['BFT_EQ_FG_DP_1'].mean():.1f} Pa
Std: {df_valid['BFT_EQ_FG_DP_1'].std():.1f} Pa"""

    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # 이미지 저장
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 차압 트렌드 그래프 저장: {output_path.absolute()}")
    print(f"   • 파일: {output_path.name}")
    print(f"   • 데이터 포인트: {len(df_valid):,}개")
    print(
        f"   • 차압 범위: {df_valid['BFT_EQ_FG_DP_1'].min():.1f} ~ {df_valid['BFT_EQ_FG_DP_1'].max():.1f} Pa"
    )
    print(f"   • 평균 차압: {df_valid['BFT_EQ_FG_DP_1'].mean():.1f} Pa")

    return output_path


def export_dp_data(
    hours=720,  # 기본 30일 (차압은 초단위로 많이 수집됨)
    output_file=None,
    measurement="SRDD",
    host="10.238.24.150",  # SRDD 개발기
    port=8086,
    username="read_user",
    password="!Skepinfluxuser25",
    database="SRDD",
):
    """
    차압 데이터를 InfluxDB에서 CSV로 내보내기 (SRDD 플랜트)

    Args:
        hours: 조회할 시간 범위 (시간 단위, 기본 720시간=30일)
        output_file: 출력 파일명 (None이면 자동 생성)
        measurement: 측정값명
        host, port, username, password, database: InfluxDB 연결 정보

    Note:
        - BFT_EQ_FG_DP_1은 초단위로 수집됨 (매우 많은 데이터)
        - 30일 조회 시 수백만 행의 데이터 예상
        - 메모리 효율을 위해 시간별 집계 후 일별 평균 계산
    """

    # 차압 관련 컬럼
    dp_columns = [
        "time",
        "_time_gateway",
        "BFT_EQ_FG_DP_1",  # 차압 센서
    ]

    print("🔍 SRDD 차압 트렌드 분석 시작")
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
        print(f"🏭 Plant: SRDD (충남)")
        print(f"🔧 Environment: Development (개발기)")
        print(f"📈 Purpose: BHI 상승 원인 분석을 위한 차압 패턴 확인")

        # 시간 범위 설정
        now = datetime.utcnow()
        start_time = now - timedelta(hours=hours)

        start_utc = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        print(
            f"📅 조회 기간: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        print(f"📊 조회 범위: {hours}시간 ({hours/24:.1f}일)")

        # 5초 간격으로 LAST 값 조회
        print(f"📌 5초 간격으로 LAST 값 조회...")

        # InfluxDB의 GROUP BY time()을 사용하여 5초 간격으로 LAST 값 조회
        query = f"""
        SELECT LAST("BFT_EQ_FG_DP_1") as "BFT_EQ_FG_DP_1"
        FROM "{measurement}" 
        WHERE time >= '{start_utc}' AND time <= '{end_utc}'
        AND "BFT_EQ_FG_DP_1" != ''
        GROUP BY time(5s)
        ORDER BY time ASC
        """

        print(f"🔎 실행 쿼리 (5초 간격):")
        print(query)
        print()

        # 데이터 조회
        print("⏳ 차압 데이터 조회 중...")
        result = client.query(query)

        # 결과를 DataFrame으로 변환
        points = list(result.get_points())
        if not points:
            print("❌ 조회된 데이터가 없습니다.")
            return None

        df = pd.DataFrame(points)

        if df.empty:
            print("❌ 조회된 데이터가 없습니다.")
            return None

        print(
            f"✅ 5초 간격 데이터 조회 완료: {len(df):,} 포인트, {len(df.columns)} 컬럼"
        )

        # 시간 컬럼 변환
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            print(f"📅 데이터 시간 범위: {df['time'].min()} ~ {df['time'].max()}")

        # _time_gateway 컬럼 추가 (time 컬럼과 동일하게 설정)
        df["_time_gateway"] = df["time"]

        # BFT_EQ_FG_DP_1 통계
        if "BFT_EQ_FG_DP_1" in df.columns:
            print(f"\n📈 차압 통계 (5초 간격):")
            total_rows = len(df)
            null_count = df["BFT_EQ_FG_DP_1"].isnull().sum()
            valid_count = df["BFT_EQ_FG_DP_1"].notna().sum()

            print(f"   - 총 포인트 수: {total_rows:,}")
            print(f"   - NULL 값: {null_count:,} ({null_count/total_rows*100:.2f}%)")
            print(f"   - 유효값: {valid_count:,} ({valid_count/total_rows*100:.2f}%)")

            if valid_count > 0:
                print(
                    f"   - 차압 범위: {df['BFT_EQ_FG_DP_1'].min():.1f} ~ {df['BFT_EQ_FG_DP_1'].max():.1f} Pa"
                )
                print(f"   - 평균 차압: {df['BFT_EQ_FG_DP_1'].mean():.1f} Pa")
                print(f"   - 표준편차: {df['BFT_EQ_FG_DP_1'].std():.1f} Pa")

        # 출력 파일명 생성
        if output_file is None:
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            output_file = f"dp_data_srdd_{timestamp}_{hours}h.csv"

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

        # 차압 트렌드 시각화
        print(f"\n📊 차압 트렌드 시각화 중...")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        plot_file = f"dp_trend_srdd_{timestamp}_{hours}h.png"
        plot_path = plot_dp_trend(df, output_file=plot_file)

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
            print(
                f"   2. git commit -m 'Add SRDD DP trend analysis: {output_path.name}'"
            )
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
        description="SRDD 차압 트렌드 분석 (BHI 상승 원인 분석용)"
    )

    parser.add_argument(
        "--hours",
        type=float,
        default=720,
        help="조회할 시간 범위 (기본: 720시간=30일)",
    )
    parser.add_argument("--output", "-o", help="출력 파일명 (기본: 자동 생성)")
    parser.add_argument(
        "--measurement", "-m", default="SRDD", help="측정값명 (기본: SRDD)"
    )

    args = parser.parse_args()

    # 차압 데이터 내보내기 실행
    result = export_dp_data(
        hours=args.hours,
        output_file=args.output,
        measurement=args.measurement,
    )

    if result:
        print(f"\n🎉 SRDD 차압 트렌드 분석 완료!")
        print(f"💡 BHI 상승 원인 분석을 위해 차압 패턴을 확인하세요.")
    else:
        print(f"\n❌ SRDD 차압 트렌드 분석 실패")


if __name__ == "__main__":
    main()
