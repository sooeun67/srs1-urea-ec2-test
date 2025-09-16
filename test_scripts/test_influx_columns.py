#!/usr/bin/env python3
"""
InfluxDB 컬럼 존재 여부 확인 스크립트
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from influxdb import InfluxDBClient


def test_column_availability():
    """요청된 컬럼들이 InfluxDB에 존재하는지 확인"""

    # 테스트할 컬럼들
    target_columns = [
        "_time_gateway",
        "ACT_SNR_PMP_BO_1",
        "ACT_SNR_PMP_BO_2",
        "ACT_SNR_PMP_BO_3",
        "ACT_SNR_PMP_BO_4",
        "ACT_SNR_PMP_BO_0",
        "SNR_NOX_PRED",
        "SNR_STATUS_CODE",
        "SNR_STAGE",
        "SNR_MESSAGE",
    ]

    print("🔍 InfluxDB 컬럼 존재 여부 확인")
    print("=" * 60)

    try:
        # InfluxDB 연결 (환경변수 또는 기본값 사용)
        host = os.getenv("INFLUX_HOST", "10.238.27.132")
        port = int(os.getenv("INFLUX_PORT", "8086"))
        username = os.getenv("INFLUX_USERNAME", "read_user")
        password = os.getenv("INFLUX_PASSWORD", "!Skepinfluxuser25")
        database = os.getenv("INFLUX_DB", "SRS1")
        measurement = os.getenv("INFLUX_MEASUREMENT", "SRS1")

        print(f"🔗 InfluxDB 연결 정보:")
        print(f"   • Host: {host}:{port}")
        print(f"   • Database: {database}")
        print(f"   • Username: {username}")
        print()

        client = InfluxDBClient(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        )

        # 최근 1시간 데이터에서 컬럼 확인
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)

        start_utc = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        print(f"📅 조회 기간: {start_utc} ~ {end_utc}")
        print(f"📊 측정값: {measurement}")
        print()

        # 1. 전체 컬럼 조회 (샘플)
        query_all = f"""
        SELECT * FROM "{measurement}" 
        WHERE time >= '{start_utc}' AND time <= '{end_utc}' 
        ORDER BY time DESC 
        LIMIT 50
        """

        print("🔎 전체 컬럼 조회 쿼리:")
        print(query_all)

        result = client.query(query_all)
        df_all = pd.DataFrame(list(result.get_points()))

        if df_all.empty:
            print("❌ 해당 기간에 데이터가 없습니다.")
            return

        print(f"✅ 전체 컬럼 수: {len(df_all.columns)}")
        print("📋 사용 가능한 모든 컬럼:")
        for i, col in enumerate(sorted(df_all.columns), 1):
            print(f"   {i:3d}. {col}")
        print()

        # 2. 요청된 컬럼들 존재 여부 확인
        print("🎯 요청된 컬럼들 존재 여부:")
        available_columns = []
        missing_columns = []

        for col in target_columns:
            if col in df_all.columns:
                available_columns.append(col)
                print(f"   ✅ {col}")
            else:
                missing_columns.append(col)
                print(f"   ❌ {col}")

        print()
        print(f"📊 요약:")
        print(f"   • 존재하는 컬럼: {len(available_columns)}/{len(target_columns)}")
        print(f"   • 누락된 컬럼: {len(missing_columns)}/{len(target_columns)}")

        if available_columns:
            # 3. 존재하는 컬럼들로 테스트 쿼리
            columns_str = ", ".join([f'"{col}"' for col in available_columns])

            query_specific = f"""
            SELECT {columns_str} FROM "{measurement}" 
            WHERE time >= '{start_utc}' AND time <= '{end_utc}' 
            ORDER BY time DESC 
            LIMIT 5
            """

            print()
            print("🔎 존재하는 컬럼들로 테스트 쿼리:")
            print(query_specific)

            result_specific = client.query(query_specific)
            df_specific = pd.DataFrame(list(result_specific.get_points()))

            if not df_specific.empty:
                print()
                print("✅ 테스트 쿼리 결과 (최근 5개 행):")
                print(df_specific)
            else:
                print("❌ 테스트 쿼리 결과가 비어있습니다.")

        if missing_columns:
            print()
            print("⚠️ 누락된 컬럼들:")
            for col in missing_columns:
                print(f"   • {col}")
            print()
            print("💡 대안 방법:")
            print("   1. 해당 컬럼들이 다른 이름으로 존재하는지 확인")
            print("   2. 해당 컬럼들이 다른 measurement에 존재하는지 확인")
            print("   3. 데이터 수집 프로세스에서 해당 컬럼들이 저장되고 있는지 확인")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_column_availability()
