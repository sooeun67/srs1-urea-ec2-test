#!/usr/bin/env python3
"""
SRS1 BFT-BHI InfluxDB 조회 및 트렌드 시각화
- 개발 InfluxDB에서 BHI 값 조회
- X축: 날짜 (UTC)
- Y축: BHI 값 (BFT_BHI_VALUE)
- 매일 1회 저장되는 BHI 값의 트렌드 시각화

실행:
cd /Users/sooeunoh/Documents/Project/2025/2.에코플랜트_소각로최적화/5.서비스화/srdd-airflow-model-main/bft-SRS1/bft-bhi/visualize
python visualize_bhi_trend_influx.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from influxdb import InfluxDBClient

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# InfluxDB 연결 정보 (SRS1 개발기)
INFLUX_HOST = os.environ.get("INFLUX_HOST", "10.238.24.150")
INFLUX_PORT = int(os.environ.get("INFLUX_PORT", "8086"))
INFLUX_USERNAME = os.environ.get("INFLUX_USERNAME", "read_user")
INFLUX_PASSWORD = os.environ.get("INFLUX_PASSWORD", "!Skepinfluxuser25")
INFLUX_DB = os.environ.get("INFLUX_DB", "SRS1")
INFLUX_MEASUREMENT = os.environ.get("INFLUX_MEASUREMENT", "SRS1")


def connect_influxdb():
    """InfluxDB 연결"""
    client = InfluxDBClient(
        host=INFLUX_HOST,
        port=INFLUX_PORT,
        username=INFLUX_USERNAME,
        password=INFLUX_PASSWORD,
        database=INFLUX_DB,
        timeout=6000,
    )

    print(f"📡 InfluxDB 연결:")
    print(f"   - Host: {INFLUX_HOST}:{INFLUX_PORT}")
    print(f"   - Database: {INFLUX_DB}")
    print(f"   - Measurement: {INFLUX_MEASUREMENT}")

    return client


def query_bhi_data(client, start_date=None, end_date=None):
    """InfluxDB에서 BHI 데이터 조회

    Args:
        client: InfluxDB 클라이언트
        start_date: 조회 시작일 (YYYY-MM-DD 형식, None이면 최근 30일)
        end_date: 조회 종료일 (YYYY-MM-DD 형식, None이면 현재)

    Returns:
        DataFrame: BHI 데이터 (time, BHI 값)
    """

    # 날짜 범위 설정
    if start_date:
        start_dt = pd.to_datetime(start_date).tz_localize("UTC")
        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        # 기본: 최근 30일
        start_str = "(now() - 30d)"

    if end_date:
        end_dt = pd.to_datetime(end_date).tz_localize("UTC")
        end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        end_str = "now()"

    # BHI 필드명 확인 필요 (예: BHI, bhi_value, BFT_BHI 등)
    # BFT_BHI_VALUE가 NULL이 아닌 행만 조회 (하루 1번만 기록되므로 효율적)
    print("\n🔍 InfluxDB에서 BHI 데이터 조회 중 (BFT_BHI_VALUE가 있는 행만)...")

    # 쿼리 작성 (BFT_BHI_VALUE가 NULL이 아닌 행만 조회)
    if start_date:
        query = f"""
        SELECT _time_gateway, BFT_BHI_CODE, BFT_BHI_DATAQUALITY, BFT_BHI_DATASTATUS, BFT_BHI_VALUE
        FROM "{INFLUX_MEASUREMENT}"
        WHERE _time_gateway >= '{start_str}' AND _time_gateway <= {end_str}
        AND BFT_BHI_VALUE IS NOT NULL
        ORDER BY _time_gateway ASC
        """
    else:
        query = f"""
        SELECT _time_gateway, BFT_BHI_CODE, BFT_BHI_DATAQUALITY, BFT_BHI_DATASTATUS, BFT_BHI_VALUE
        FROM "{INFLUX_MEASUREMENT}"
        WHERE _time_gateway >= {start_str} AND _time_gateway <= {end_str}
        AND BFT_BHI_VALUE IS NOT NULL
        ORDER BY _time_gateway ASC
        """

    print(f"📝 쿼리:")
    print(query)

    try:
        result = client.query(query)
        points = list(result.get_points()) if result else []

        if not points:
            print("❌ 데이터가 없습니다.")
            return pd.DataFrame()

        df = pd.DataFrame(points)
        print(f"✅ 조회 성공: {len(df)}행")

        # 컬럼 확인
        print(f"\n📋 조회된 컬럼: {list(df.columns)}")

        # BHI 관련 컬럼 찾기
        bhi_cols = [
            col for col in df.columns if "bhi" in col.lower() or "health" in col.lower()
        ]
        if bhi_cols:
            print(f"🎯 BHI 관련 컬럼: {bhi_cols}")
        else:
            print("⚠️ BHI 관련 컬럼을 찾을 수 없습니다.")
            print("📊 샘플 데이터 (처음 3행):")
            print(df.head(3))

        return df

    except Exception as e:
        print(f"❌ 쿼리 실패: {e}")
        return pd.DataFrame()


def extract_bhi_values(df):
    """DataFrame에서 BHI 값 추출

    Args:
        df: InfluxDB 조회 결과 DataFrame

    Returns:
        DataFrame: time과 BHI 값만 포함
    """
    if df.empty:
        return pd.DataFrame()

    # BHI 관련 컬럼 정의
    bhi_value_col = "BFT_BHI_VALUE"
    bhi_related_cols = [
        "_time_gateway",
        "BFT_BHI_CODE",
        "BFT_BHI_DATAQUALITY",
        "BFT_BHI_DATASTATUS",
        "BFT_BHI_VALUE",
    ]

    # BFT_BHI_VALUE 컬럼 확인
    if bhi_value_col not in df.columns:
        print(f"❌ {bhi_value_col} 컬럼을 찾을 수 없습니다.")
        print(f"📋 사용 가능한 컬럼: {list(df.columns)}")
        return pd.DataFrame()

    print(f"✅ BHI 컬럼 발견: {bhi_value_col}")

    # 존재하는 BHI 관련 컬럼만 선택
    available_cols = [col for col in bhi_related_cols if col in df.columns]

    print(f"✅ 추출할 컬럼: {available_cols}")

    # BHI 관련 컬럼 추출
    df_bhi = df[available_cols].copy()

    # BHI 컬럼 추가 (시각화용)
    df_bhi["BHI"] = df_bhi[bhi_value_col]

    # _time_gateway를 datetime으로 변환 (UTC 기준 유지)
    df_bhi["_time_gateway"] = pd.to_datetime(df_bhi["_time_gateway"], utc=True)

    # BHI 값이 유효한 행만 필터링
    df_bhi = df_bhi[df_bhi["BHI"].notna()].copy()

    # 날짜별로 정렬
    df_bhi = df_bhi.sort_values("_time_gateway").reset_index(drop=True)

    print(f"\n📊 BHI 데이터 추출 완료:")
    print(f"   - 전체 행 수: {len(df_bhi)}")
    print(
        f"   - 시간 범위(UTC): {df_bhi['_time_gateway'].min()} ~ {df_bhi['_time_gateway'].max()}"
    )
    print(f"   - BHI 범위: {df_bhi['BHI'].min():.2f} ~ {df_bhi['BHI'].max():.2f}")

    # 샘플 데이터 출력 (처음 5개)
    print("\n📋 샘플 데이터 (처음 5개):")
    sample_cols = [
        col
        for col in [
            "_time_gateway",
            "BFT_BHI_CODE",
            "BFT_BHI_DATAQUALITY",
            "BFT_BHI_DATASTATUS",
            "BFT_BHI_VALUE",
        ]
        if col in df_bhi.columns
    ]
    if sample_cols:
        print(df_bhi[sample_cols].head())

    return df_bhi


def aggregate_daily_bhi(df_bhi):
    """BHI 데이터를 일별로 집계

    Args:
        df_bhi: _time_gateway와 BHI 값이 있는 DataFrame

    Returns:
        DataFrame: 일별 BHI 평균/최신값
    """
    if df_bhi.empty:
        return pd.DataFrame()

    # 날짜별로 그룹화 (UTC 기준)
    df_bhi["date"] = df_bhi["_time_gateway"].dt.date

    # 매일 1회 저장된다고 가정 -> 각 날짜의 마지막 값 사용
    df_daily = (
        df_bhi.groupby("date")
        .agg(
            {
                "BHI": "last",  # 마지막 값
                "_time_gateway": "last",  # 마지막 시간 (UTC)
            }
        )
        .reset_index()
    )

    print(f"\n📅 일별 BHI 집계:")
    print(f"   - 총 {len(df_daily)}일")
    print(f"   - 기간(UTC): {df_daily['date'].min()} ~ {df_daily['date'].max()}")

    return df_daily


def plot_bhi_trend(df_daily, save_path="bhi_trend_srs1.png"):
    """BHI 트렌드 시각화

    Args:
        df_daily: 일별 BHI 데이터
        save_path: 저장 경로
    """
    if df_daily.empty:
        print("❌ 시각화할 데이터가 없습니다.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # BHI 트렌드 라인
    ax.plot(
        df_daily["date"],
        df_daily["BHI"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="steelblue",
        label="BHI (BFT Health Index)",
    )

    # 기준선 추가
    ax.axhline(
        y=80,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="주의 (80)",
    )
    ax.axhline(
        y=90,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="교체 권장 (90)",
    )

    # 평균선
    mean_bhi = df_daily["BHI"].mean()
    ax.axhline(
        y=mean_bhi,
        color="green",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label=f"평균 ({mean_bhi:.1f})",
    )

    # 그래프 설정
    ax.set_xlabel("날짜 (UTC)", fontsize=12)
    ax.set_ylabel("BHI 값 (%)", fontsize=12)
    ax.set_title(
        f"SRS1 백필터 건강도 지수 (BHI) 트렌드\n"
        f'기간(UTC): {df_daily["date"].min()} ~ {df_daily["date"].max()} ({len(df_daily)}일)',
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    # Y축 범위 설정
    y_min = max(0, df_daily["BHI"].min() - 10)
    y_max = min(100, df_daily["BHI"].max() + 10)
    ax.set_ylim(y_min, y_max)

    # X축 날짜 포맷
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df_daily) // 20)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ 그래프 저장: {save_path}")


def print_summary_table(df_daily):
    """BHI 데이터 요약 테이블 출력"""
    if df_daily.empty:
        return

    print("\n" + "=" * 70)
    print("📊 SRS1 BHI 트렌드 요약")
    print("=" * 70)

    # 기본 통계
    print(f"\n📈 기본 통계:")
    print(f"   - 총 일수: {len(df_daily)}일")
    print(f"   - 기간: {df_daily['date'].min()} ~ {df_daily['date'].max()}")
    print(f"   - 평균 BHI: {df_daily['BHI'].mean():.2f}%")
    print(
        f"   - 최소 BHI: {df_daily['BHI'].min():.2f}% ({df_daily.loc[df_daily['BHI'].idxmin(), 'date']})"
    )
    print(
        f"   - 최대 BHI: {df_daily['BHI'].max():.2f}% ({df_daily.loc[df_daily['BHI'].idxmax(), 'date']})"
    )
    print(f"   - 표준편차: {df_daily['BHI'].std():.2f}%")

    # 최근 7일 평균
    if len(df_daily) >= 7:
        recent_7d = df_daily.tail(7)["BHI"].mean()
        print(f"   - 최근 7일 평균: {recent_7d:.2f}%")

    # 최근 30일 평균
    if len(df_daily) >= 30:
        recent_30d = df_daily.tail(30)["BHI"].mean()
        print(f"   - 최근 30일 평균: {recent_30d:.2f}%")

    # 경고 수준 확인
    warning_count = (df_daily["BHI"] >= 80).sum()
    critical_count = (df_daily["BHI"] >= 90).sum()
    print(f"\n⚠️ 경고 수준:")
    print(
        f"   - BHI ≥ 80% (주의): {warning_count}일 ({warning_count/len(df_daily)*100:.1f}%)"
    )
    print(
        f"   - BHI ≥ 90% (교체 권장): {critical_count}일 ({critical_count/len(df_daily)*100:.1f}%)"
    )

    # 최근 10일 데이터
    print(f"\n📅 최근 10일 BHI 값:")
    print("=" * 70)
    print(f"{'날짜 (UTC)':<15} {'BHI (%)':<10} {'상태':<15} {'시간 (UTC)':<25}")
    print("=" * 70)

    for _, row in df_daily.tail(10).iterrows():
        bhi = row["BHI"]
        if bhi >= 90:
            status = "🔴 교체 권장"
        elif bhi >= 80:
            status = "🟠 주의"
        else:
            status = "🟢 정상"

        print(
            f"{row['date']!s:<15} {bhi:>7.2f}    {status:<15} {row['_time_gateway']!s:<25}"
        )

    print("=" * 70)


def main():
    print("🚀" + "=" * 68)
    print("🚀 SRS1 BFT-BHI InfluxDB 조회 및 트렌드 시각화")
    print("🚀" + "=" * 68)

    # 1) InfluxDB 연결
    print("\n📡 InfluxDB 연결 중...")
    client = connect_influxdb()

    # 2) BHI 데이터 조회 (최근 30일)
    # 특정 기간 조회 시: start_date='2025-09-01', end_date='2025-10-13'
    df_raw = query_bhi_data(client, start_date=None, end_date=None)

    if df_raw.empty:
        print("❌ 데이터 조회 실패")
        return

    # 3) BHI 값 추출
    df_bhi = extract_bhi_values(df_raw)

    if df_bhi.empty:
        print("❌ BHI 값 추출 실패")
        return

    # 4) 일별 집계
    df_daily = aggregate_daily_bhi(df_bhi)

    if df_daily.empty:
        print("❌ 일별 집계 실패")
        return

    # 5) 요약 테이블 출력
    print_summary_table(df_daily)

    # 6) 트렌드 시각화
    print("\n📊 BHI 트렌드 시각화 중...")
    plot_bhi_trend(df_daily, save_path="bhi_trend_srs1.png")

    # 7) CSV 저장 (원본 BHI 데이터 - 모든 BHI 관련 컬럼 포함)
    csv_path = "bhi_daily_srs1.csv"

    # df_bhi에서 원본 컬럼들 저장 (_time_gateway, BFT_BHI_*)
    bhi_save_cols = [
        col
        for col in df_bhi.columns
        if col
        in [
            "_time_gateway",
            "BFT_BHI_CODE",
            "BFT_BHI_DATAQUALITY",
            "BFT_BHI_DATASTATUS",
            "BFT_BHI_VALUE",
            "BHI",
        ]
    ]
    df_bhi[bhi_save_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ CSV 저장: {csv_path}")
    print(f"   저장된 컬럼: {bhi_save_cols}")

    print("\n" + "=" * 70)
    print("✅ SRS1 BHI 트렌드 분석 완료!")
    print("=" * 70)
    print(f"📊 생성된 파일:")
    print(f"   1. bhi_trend_srs1.png - BHI 트렌드 그래프")
    print(f"   2. bhi_daily_srs1.csv - 일별 BHI 데이터")
    print("=" * 70)


if __name__ == "__main__":
    main()
