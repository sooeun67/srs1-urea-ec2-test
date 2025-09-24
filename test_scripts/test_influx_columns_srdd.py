#!/usr/bin/env python3
"""
SRDD InfluxDB 컬럼 확인 스크립트
- SRDD 데이터베이스에서 특정 컬럼들의 존재 여부 및 데이터 품질 확인
- 현재 시각 기준 이전 20개 행 출력

[0918] SRDD 사이트용 컬럼 확인 스크립트 생성
"""

import os
import sys
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient
import pandas as pd


def check_srdd_influxdb_columns():
    """SRDD InfluxDB 컬럼 확인"""

    # InfluxDB 연결 설정
    host = os.getenv("INFLUX_HOST", "10.238.24.150")
    port = int(os.getenv("INFLUX_PORT", "8086"))
    username = os.getenv("INFLUX_USERNAME", "read_user")
    password = os.getenv("INFLUX_PASSWORD", "!Skepinfluxuser25")
    database = os.getenv("INFLUX_DB", "SRDD")
    measurement = os.getenv("INFLUX_MEASUREMENT", "SRDD")

    print("🔍 SRDD InfluxDB 컬럼 확인 시작")
    print(f"   📍 호스트: {host}:{port}")
    print(f"   🗄️ 데이터베이스: {database}")
    print(f"   📊 측정값: {measurement}")

    try:
        # InfluxDB 클라이언트 생성
        client = InfluxDBClient(
            url=f"http://{host}:{port}",
            token=f"{username}:{password}",
            org="-",
            timeout=30000,
        )

        # 현재 시각 기준 이전 1분 데이터 조회
        now = datetime.utcnow()
        start_time = now - timedelta(minutes=1)

        start_utc = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        print(f"   📅 조회 시간: {start_utc} ~ {end_utc}")

        # 1. 전체 컬럼 조회
        print("\n📊 1. 전체 컬럼 조회")
        query_all = f"""
        SELECT * FROM "{measurement}" 
        WHERE time >= '{start_utc}' AND time <= '{end_utc}'
        ORDER BY time DESC 
        LIMIT 50
        """

        result_all = client.query_api().query_data_frame(query_all)

        if result_all.empty:
            print("   ⚠️ 조회된 데이터가 없습니다.")
            return

        print(f"   ✅ 조회된 데이터: {len(result_all)}행")
        print(f"   📋 전체 컬럼 수: {len(result_all.columns)}개")
        print(f"   📋 컬럼 목록: {list(result_all.columns)}")

        # 2. SRDD 특정 컬럼들 확인
        print("\n🎯 2. SRDD 특정 컬럼 확인")

        # 확인할 컬럼들 (SRDD 기본 센서 컬럼들만)
        target_columns = [
            "_time_gateway",
            "BR1_EO_FG_A",  # 보일러 출구 연소가스 농도
            "SNR_PMP_UW_S_1",  # 실제 요소수 펌프 Hz
            "ICF_SCS_FG_T_1",  # 노 출구 온도
            "ICF_TMS_NOX_A",  # 보정 전 NOx
            "ACT_STATUS",  # 상태 코드
        ]

        # 존재하는 컬럼들 확인
        existing_columns = [col for col in target_columns if col in result_all.columns]
        missing_columns = [
            col for col in target_columns if col not in result_all.columns
        ]

        print(f"   ✅ 존재하는 컬럼 ({len(existing_columns)}개):")
        for col in existing_columns:
            non_null_count = result_all[col].notna().sum()
            print(f"      - {col}: {non_null_count}개 데이터")

        if missing_columns:
            print(f"   ❌ 누락된 컬럼 ({len(missing_columns)}개):")
            for col in missing_columns:
                print(f"      - {col}")

        # 3. 특정 컬럼들의 최근 20개 행 출력
        if existing_columns:
            print(f"\n📋 3. 최근 20개 행 출력 (존재하는 컬럼들)")

            columns_str = ", ".join(existing_columns)
            query_specific = f"""
            SELECT {columns_str} FROM "{measurement}" 
            WHERE time >= '{start_utc}' AND time <= '{end_utc}'
            ORDER BY time DESC 
            LIMIT 20
            """

            result_specific = client.query_api().query_data_frame(query_specific)

            if not result_specific.empty:
                print(f"   📊 조회된 행 수: {len(result_specific)}")
                print(
                    f"   📅 시간 범위: {result_specific['time'].min()} ~ {result_specific['time'].max()}"
                )

                # 데이터 출력 (처음 5행만)
                print(f"\n   📋 데이터 샘플 (처음 5행):")
                pd.set_option("display.max_columns", None)
                pd.set_option("display.width", None)
                pd.set_option("display.max_colwidth", 20)
                print(result_specific.head().to_string(index=False))
            else:
                print("   ⚠️ 특정 컬럼 조회 결과가 없습니다.")

        # 4. 데이터 품질 요약
        print(f"\n📊 4. 데이터 품질 요약")
        print(f"   📈 전체 행 수: {len(result_all)}")
        print(
            f"   📅 시간 범위: {result_all['time'].min()} ~ {result_all['time'].max()}"
        )

        # 각 컬럼별 데이터 품질
        for col in existing_columns:
            if col in result_all.columns:
                total_count = len(result_all)
                non_null_count = result_all[col].notna().sum()
                null_count = total_count - non_null_count
                null_ratio = (null_count / total_count) * 100 if total_count > 0 else 0

                print(f"   📊 {col}:")
                print(
                    f"      - 전체: {total_count}개, 유효: {non_null_count}개, 누락: {null_count}개 ({null_ratio:.1f}%)"
                )

        client.close()
        print(f"\n✅ SRDD InfluxDB 컬럼 확인 완료")

    except Exception as e:
        print(f"❌ InfluxDB 연결 또는 쿼리 실패: {e}")


if __name__ == "__main__":
    check_srdd_influxdb_columns()
