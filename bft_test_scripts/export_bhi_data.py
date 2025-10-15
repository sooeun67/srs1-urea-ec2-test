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

# 프로젝트 루트 디렉토리를 Python 경로에 추가
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influxdb import InfluxDBClient


def export_bhi_data(
    hours=24,  # 기본 1일 (초단위 수집으로 1일=86,400행)
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
        hours: 조회할 시간 범위 (시간 단위, 기본 24시간=1일)
        output_file: 출력 파일명 (None이면 자동 생성)
        measurement: 측정값명
        host, port, username, password, database: InfluxDB 연결 정보

    Note:
        InfluxDB는 초단위로 데이터 수집
        - 1시간 = 3,600행
        - 1일 = 86,400행
        - 30일 = 2,592,000행 (조회 시간 오래 걸림)
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

        # 쿼리 생성 (BFT-BHI 컬럼만)
        columns_str = ", ".join([f'"{col}"' for col in bhi_columns])
        query = f"""
        SELECT {columns_str} FROM "{measurement}" 
        WHERE time >= '{start_utc}' AND time <= '{end_utc}'
        ORDER BY time ASC
        """

        print(f"\n🔎 실행 쿼리:")
        print(query)
        print()

        # 데이터 조회
        print("⏳ 데이터 조회 중...")
        result = client.query(query)
        df = pd.DataFrame(list(result.get_points()))

        if df.empty:
            print("❌ 조회된 데이터가 없습니다.")
            return None

        print(f"✅ 조회 완료: {len(df):,} 행, {len(df.columns)} 컬럼")

        # 시간 컬럼 변환
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            print(f"📅 데이터 시간 범위: {df['time'].min()} ~ {df['time'].max()}")

        # BFT_BHI_VALUE 통계
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

        # GitHub을 통한 데이터 전송 안내
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n🔄 GitHub을 통한 데이터 전송 방법:")

        if file_size_mb > 25:
            print(f"   ⚠️ 파일이 {file_size_mb:.1f}MB로 GitHub 제한(25MB)을 초과합니다.")
            print(f"   💡 --hours 옵션으로 데이터 범위를 줄이세요.")
        else:
            print(f"   1. git add {output_path.name}")
            print(f"   2. git commit -m 'Add BHI data: {output_path.name}'")
            print(f"   3. git push origin main")
            print(f"   4. 로컬에서 git pull origin main 후 {output_path.name} 사용")

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
        default=24,
        help="조회할 시간 범위 (기본: 24시간=1일, 1시간=3,600행)",
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
