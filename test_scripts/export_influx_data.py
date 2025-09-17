#!/usr/bin/env python3
"""
InfluxDB 데이터를 CSV로 내보내는 스크립트
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


def export_influx_data(
    columns=None,
    hours=1,
    output_file=None,
    measurement="SRS1",
    host="10.238.24.150",
    port=8086,
    username="read_user",
    password="!Skepinfluxuser25",
    database="SRS1",
):
    """
    InfluxDB 데이터를 CSV로 내보내기

    Args:
        columns: 내보낼 컬럼 리스트 (None이면 모든 컬럼)
        hours: 조회할 시간 범위 (시간 단위)
        output_file: 출력 파일명 (None이면 자동 생성)
        measurement: 측정값명
        host, port, username, password, database: InfluxDB 연결 정보
    """

    print("🔍 InfluxDB 데이터 내보내기 시작")
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

        # 시간 범위 설정
        now = datetime.utcnow()
        start_time = now - timedelta(hours=hours)

        start_utc = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        print(
            f"📅 조회 기간: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        print(f"📊 조회 범위: {hours}시간")

        # 쿼리 생성
        if columns:
            columns_str = ", ".join([f'"{col}"' for col in columns])
            query = f"""
            SELECT {columns_str} FROM "{measurement}" 
            WHERE time >= '{start_utc}' AND time <= '{end_utc}'
            ORDER BY time ASC
            """
        else:
            query = f"""
            SELECT * FROM "{measurement}" 
            WHERE time >= '{start_utc}' AND time <= '{end_utc}'
            ORDER BY time ASC
            """

        print(f"🔎 실행 쿼리:")
        print(query)
        print()

        # 데이터 조회
        print("⏳ 데이터 조회 중...")
        result = client.query(query)
        df = pd.DataFrame(list(result.get_points()))

        if df.empty:
            print("❌ 조회된 데이터가 없습니다.")
            return None

        print(f"✅ 조회 완료: {len(df)} 행, {len(df.columns)} 컬럼")

        # 시간 컬럼 변환
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            print(f"📅 데이터 시간 범위: {df['time'].min()} ~ {df['time'].max()}")

        # 출력 파일명 생성
        if output_file is None:
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            if columns:
                col_suffix = (
                    f"_{'_'.join(columns[:3])}"
                    if len(columns) <= 3
                    else f"_{len(columns)}cols"
                )
            else:
                col_suffix = "_all"
            output_file = f"influx_data_{timestamp}{col_suffix}.csv"

        # CSV 저장
        output_path = Path(output_file)
        df.to_csv(output_path, index=False)

        print(f"💾 CSV 저장 완료: {output_path.absolute()}")
        print(f"📊 파일 크기: {output_path.stat().st_size / 1024:.1f} KB")

        # 데이터 미리보기
        print("\n📋 데이터 미리보기 (처음 5행):")
        print(df.head())

        print(f"\n✅ 내보내기 완료!")
        print(f"   • 파일: {output_path.name}")
        print(f"   • 경로: {output_path.absolute()}")
        print(f"   • 행 수: {len(df):,}")
        print(f"   • 컬럼 수: {len(df.columns)}")

        return output_path

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="InfluxDB 데이터를 CSV로 내보내기")

    parser.add_argument(
        "--columns",
        "-c",
        nargs="+",
        help="내보낼 컬럼명 (예: time _time_gateway ACT_SNR_PMP_BO_1)",
    )
    parser.add_argument(
        "--hours", "-h", type=float, default=1, help="조회할 시간 범위 (기본: 1시간)"
    )
    parser.add_argument("--output", "-o", help="출력 파일명 (기본: 자동 생성)")
    parser.add_argument(
        "--measurement", "-m", default="SRS1", help="측정값명 (기본: SRS1)"
    )

    # 미리 정의된 컬럼 세트
    parser.add_argument(
        "--pump-columns", action="store_true", help="펌프 관련 컬럼만 내보내기"
    )
    parser.add_argument(
        "--prediction-columns", action="store_true", help="예측 관련 컬럼만 내보내기"
    )
    parser.add_argument(
        "--all-target-columns", action="store_true", help="모든 타겟 컬럼 내보내기"
    )

    args = parser.parse_args()

    # 컬럼 세트 정의
    if args.pump_columns:
        columns = [
            "time",
            "_time_gateway",
            "ACT_SNR_PMP_BO_0",
            "ACT_SNR_PMP_BO_1",
            "ACT_SNR_PMP_BO_2",
            "ACT_SNR_PMP_BO_3",
            "ACT_SNR_PMP_BO_4",
            "SNR_PMP_UW_S_1",
        ]
    elif args.prediction_columns:
        columns = [
            "time",
            "_time_gateway",
            "SNR_NOX_PRED",
            "SNR_STATUS_CODE",
            "SNR_STAGE",
            "SNR_MESSAGE",
            "ICF_TMS_NOX_A",
            "BR1_EO_O2_A",
        ]
    elif args.all_target_columns:
        columns = [
            "time",
            "_time_gateway",
            "ACT_SNR_PMP_BO_0",
            "ACT_SNR_PMP_BO_1",
            "ACT_SNR_PMP_BO_2",
            "ACT_SNR_PMP_BO_3",
            "ACT_SNR_PMP_BO_4",
            "SNR_NOX_PRED",
            "SNR_STATUS_CODE",
            "SNR_STAGE",
            "SNR_MESSAGE",
            "SNR_PMP_UW_S_1",
            "BR1_EO_O2_A",
            "ICF_CCS_FG_T_1",
            "ICF_SCS_FG_T_1",
            "ICF_TMS_NOX_A",
            "ACC_SNR_AI_1A",
            "ACT_STATUS",
        ]
    else:
        columns = args.columns

    # 데이터 내보내기 실행
    result = export_influx_data(
        columns=columns,
        hours=args.hours,
        output_file=args.output,
        measurement=args.measurement,
    )

    if result:
        print(f"\n🚀 다음 명령으로 로컬로 다운로드하세요:")
        print(
            f"scp -i ~/.ssh/your-key.pem ssm-user@your-ec2-ip:~/urea/sooeun/srs1-urea-ec2-test/{result.name} ./"
        )


if __name__ == "__main__":
    main()
