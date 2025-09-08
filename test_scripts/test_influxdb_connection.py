"""
InfluxDB 연결 테스트 및 실시간 데이터 조회
"""

import os
import sys

sys.path.append("..")

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
from datetime import datetime, timedelta

# InfluxDB 설정 (기존 aws 코드 참고)
INFLUXDB_HOST = "10.238.27.132"
INFLUXDB_PORT = "8086"
INFLUXDB_USERNAME = "read_user"
INFLUXDB_PASSWORD = "!Skepinfluxuser25"
INFLUXDB_DATABASE = "SRS1"  # 데이터베이스명
INFLUXDB_TIMEOUT = 30
INFLUXDB_BUCKET = "SRS1"
INFLUXDB_MEASUREMENT = "SRS1"  # 실제 측정값(테이블) 이름


def test_influxdb_connection():
    """InfluxDB 연결을 테스트합니다."""
    print("=" * 50)
    print("InfluxDB 연결 테스트 시작")
    print("=" * 50)

    try:
        # InfluxDB 클라이언트 생성 (기존 aws 코드 방식)
        from influxdb import InfluxDBClient as LegacyInfluxDBClient

        client = LegacyInfluxDBClient(
            host=INFLUXDB_HOST,
            port=INFLUXDB_PORT,
            username=INFLUXDB_USERNAME,
            password=INFLUXDB_PASSWORD,
            database=INFLUXDB_DATABASE,
            timeout=INFLUXDB_TIMEOUT,
        )

        # 연결 테스트
        print("InfluxDB 연결 테스트...")

        # 데이터베이스 목록 조회
        databases = client.get_list_database()
        print(f"사용 가능한 데이터베이스: {[db['name'] for db in databases]}")

        # 현재 데이터베이스의 측정값 목록 조회
        measurements = client.get_list_measurements()
        print(f"현재 데이터베이스의 측정값: {[m['name'] for m in measurements]}")

        return client

    except Exception as e:
        print(f"InfluxDB 연결 실패: {e}")
        return None


def test_realtime_data_query(client):
    """실시간 데이터 조회를 테스트합니다."""
    if client is None:
        print("클라이언트가 없어서 데이터 조회를 건너뜁니다.")
        return None

    print("\n" + "=" * 50)
    print("실시간 데이터 조회 테스트")
    print("=" * 50)

    try:
        # 최근 1시간 데이터 조회 (측정값: SRS1)
        query = f"""
        SELECT * FROM "{INFLUXDB_MEASUREMENT}" 
        WHERE time >= now() - 1h 
        ORDER BY time DESC 
        LIMIT 10
        """

        print(f"쿼리 실행: {query}")
        result = client.query(query)

        points = list(result.get_points()) if result else []
        point_count = len(points)
        print(f"조회된 데이터 포인트 수: {point_count}")

        if point_count == 0:
            print("조회된 데이터가 없습니다.")
            return None

        df = pd.DataFrame(points)
        print(f"데이터프레임 형태: {df.shape}")
        print("최신 데이터 샘플:")
        print(df.head())
        return df

    except Exception as e:
        print(f"데이터 조회 실패: {e}")
        return None


if __name__ == "__main__":
    # InfluxDB 연결 테스트
    client = test_influxdb_connection()

    # 실시간 데이터 조회 테스트
    if client:
        data = test_realtime_data_query(client)

        if data is not None:
            print("\n✅ InfluxDB 연결 및 데이터 조회 성공!")
        else:
            print("\n⚠️ InfluxDB 연결은 성공했지만 데이터 조회 실패")
    else:
        print("\n❌ InfluxDB 연결 실패")
