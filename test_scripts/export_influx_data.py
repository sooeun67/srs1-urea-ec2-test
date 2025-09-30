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
    host="10.238.27.132", # 운영
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

        # GitHub을 통한 데이터 전송 안내
        print(f"\n🔄 GitHub을 통한 데이터 전송 방법:")
        print(f"   1. git add {output_path.name}")
        print(f"   2. git commit -m 'Add exported data: {output_path.name}'")
        print(f"   3. git push origin main")
        print(f"   4. 로컬에서 git pull origin main 후 {output_path.name} 사용")

        # 파일 크기 확인
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 25:
            print(f"   ⚠️ 파일이 {file_size_mb:.1f}MB로 GitHub 제한(25MB)을 초과합니다.")
            print(
                f"   💡 --hours 옵션으로 데이터 범위를 줄이거나 특정 컬럼만 선택하세요."
            )

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
        "--hours", type=float, default=1, help="조회할 시간 범위 (기본: 1시간)"
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

    # 데이터 분할 옵션
    parser.add_argument(
        "--split-hours",
        type=float,
        help="데이터를 지정된 시간 단위로 분할 (예: 0.5 = 30분씩)",
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
            "NOX_30m_Value",
            "NOX_EQ_Status",
            "NOX_Value",
            "SNR_EQ_UW_F_1",
            "ACT_SNR_PMP_UW_S",
        ]
    else:
        columns = args.columns

    # 데이터 내보내기 실행
    if args.split_hours and args.hours > args.split_hours:
        # 데이터 분할 처리
        total_hours = args.hours
        split_hours = args.split_hours
        num_splits = int(total_hours / split_hours) + (
            1 if total_hours % split_hours > 0 else 0
        )

        print(
            f"📊 데이터 분할: {total_hours}시간을 {split_hours}시간씩 {num_splits}개 파일로 분할"
        )

        results = []
        for i in range(num_splits):
            start_offset = i * split_hours
            end_offset = min((i + 1) * split_hours, total_hours)
            current_hours = end_offset - start_offset

            # 분할된 파일명 생성
            if args.output:
                base_name = args.output.rsplit(".", 1)[0]
                ext = args.output.rsplit(".", 1)[1] if "." in args.output else "csv"
                split_output = f"{base_name}_part{i+1:02d}.{ext}"
            else:
                split_output = None

            print(
                f"\n📁 파트 {i+1}/{num_splits}: 최근 {end_offset:.1f}~{start_offset:.1f}시간 전 데이터"
            )

            result = export_influx_data(
                columns=columns,
                hours=current_hours,
                output_file=split_output,
                measurement=args.measurement,
            )

            if result:
                results.append(result)

        if results:
            print(f"\n🎉 총 {len(results)}개 파일 생성 완료!")
            print(f"🔄 GitHub 업로드 명령:")
            for result in results:
                print(f"   git add {result.name}")
            print(f"   git commit -m 'Add exported data: {len(results)} files'")
            print(f"   git push origin main")
    else:
        # 단일 파일 처리
        result = export_influx_data(
            columns=columns,
            hours=args.hours,
            output_file=args.output,
            measurement=args.measurement,
        )


if __name__ == "__main__":
    main()
