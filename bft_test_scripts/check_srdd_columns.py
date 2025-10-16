#!/usr/bin/env python3
"""
SRDD 데이터베이스의 컬럼 확인 스크립트
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


def check_srdd_columns():
    """SRDD 데이터베이스의 컬럼 구조 확인"""

    print("🔍 SRDD 데이터베이스 컬럼 구조 확인")
    print("=" * 60)

    try:
        # InfluxDB 연결
        client = InfluxDBClient(
            host="10.238.24.150",
            port=8086,
            username="read_user",
            password="!Skepinfluxuser25",
            database="SRDD",
        )

        print(f"🔗 InfluxDB 연결: 10.238.24.150:8086/SRDD")
        print(f"📊 Measurement: SRDD")

        # 최근 1시간 데이터 조회하여 컬럼 확인
        now = datetime.utcnow()
        start_time = now - timedelta(hours=1)
        start_utc = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        print(
            f"📅 조회 기간: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

        # 1. 최근 데이터 샘플 조회
        print(f"\n🔎 최근 데이터 샘플 조회...")
        query1 = f"""
        SELECT * FROM "SRDD" 
        WHERE time >= '{start_utc}' AND time <= '{end_utc}'
        LIMIT 10
        """

        print(f"쿼리: {query1}")
        result1 = client.query(query1)
        points1 = list(result1.get_points())

        if points1:
            df1 = pd.DataFrame(points1)
            print(f"✅ 샘플 데이터 조회 성공: {len(df1)} 행")
            print(f"📋 컬럼 목록:")
            for i, col in enumerate(df1.columns, 1):
                print(f"   {i:2d}. {col}")

            print(f"\n📊 샘플 데이터 (처음 3행):")
            print(df1.head(3))
        else:
            print("❌ 샘플 데이터가 없습니다.")

        # 2. 차압 관련 컬럼 확인
        print(f"\n🔎 차압 관련 컬럼 검색...")

        # 가능한 차압 컬럼명들
        dp_columns = [
            "BFT_EQ_FG_DP" "BFT_EQ_FG_DP_1",
            "BFT_DP_1",
            "BFT_DIFF_PRESSURE_1",
            "DP_1",
            "DIFF_PRESSURE_1",
            "FG_DP_1",
            "BAG_FILTER_DP_1",
            "BF_DP_1",
        ]

        for dp_col in dp_columns:
            query2 = f"""
            SELECT "{dp_col}" FROM "SRDD" 
            WHERE time >= '{start_utc}' AND time <= '{end_utc}'
            AND "{dp_col}" != ''
            LIMIT 5
            """

            try:
                result2 = client.query(query2)
                points2 = list(result2.get_points())

                if points2:
                    print(f"✅ '{dp_col}' 컬럼 존재 - {len(points2)}개 값")
                    df2 = pd.DataFrame(points2)
                    if dp_col in df2.columns:
                        non_null_count = df2[dp_col].notna().sum()
                        print(f"   - 유효값: {non_null_count}/{len(df2)}")
                        if non_null_count > 0:
                            print(
                                f"   - 값 범위: {df2[dp_col].min()} ~ {df2[dp_col].max()}"
                            )
                else:
                    print(f"❌ '{dp_col}' 컬럼 없음 또는 데이터 없음")

            except Exception as e:
                print(f"❌ '{dp_col}' 쿼리 오류: {e}")

        # 3. 모든 컬럼의 데이터 타입 확인
        print(f"\n🔎 데이터 타입 확인...")
        query3 = f"""
        SHOW FIELD KEYS FROM "SRDD"
        """

        try:
            result3 = client.query(query3)
            print(f"✅ 필드 키 조회 성공")
            for series in result3:
                print(f"📋 필드 목록:")
                for point in series:
                    field_name = point.get("fieldKey", "")
                    field_type = point.get("fieldType", "")
                    if field_name:
                        print(f"   - {field_name}: {field_type}")
        except Exception as e:
            print(f"❌ 필드 키 조회 오류: {e}")

        return True

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    check_srdd_columns()
