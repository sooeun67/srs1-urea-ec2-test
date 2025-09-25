#!/usr/bin/env python3
"""
SRDD 사이트용 MLflow 추론 테스트 스크립트
- InfluxDB에서 실시간 데이터 조회
- 5초 윈도우로 10분간 데이터 요약 (120개 행)
- GP 모델과 LGBM 모델을 통한 NOx 예측 및 Hz 추천
- PumpOptimizer를 통한 최적 Hz 결정

[0918] SRDD 사이트용 스크립트 생성
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import project modules
from config.column_config import ColumnConfig
from config.preprocessing_config import (
    InferPreprocessingConfig,
    LGBMInferPreprocessingConfig,
)
from config.model_config import GPModelConfig, LGBMModelConfig
from config.optimization_config import OptimizationConfig
from config.rule_config import RuleConfig
from src.data_processing.preprocessor import Preprocessor, LGBMFeaturePreprocessor
from src.models.gaussian_process import GaussianProcessNOxModel
from src.models.lgbm import LGBMNOxModel
from src.optimization.pump_optimizer import PumpOptimizer
from src.optimization.pump_hz_adjuster import LGBMPumpHzAdjuster
from utils.logger import LoggerConfig

# InfluxDB client
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS


def aggregate_10min_to_5s(
    df: pd.DataFrame, preprocessor: Preprocessor, cc: ColumnConfig
) -> pd.DataFrame:
    """최근 10분 데이터를 5초 윈도우로 요약하여 120행 반환.

    새로운 전처리 파이프라인을 활용하여 ffill 보간을 수행합니다.

    - 센서 컬럼: 5초 평균
    - *_status 컬럼: 각 윈도우의 마지막 값
    - _time_gateway: 각 윈도우의 경계 시각(오른쪽 라벨)
    """
    if "time" not in df.columns:
        raise KeyError("Influx 응답에 'time' 컬럼이 없습니다.")

    # 1) InfluxDB 컬럼명을 ColumnConfig 컬럼명으로 매핑
    df_mapped = df.copy()
    df_mapped[cc.col_datetime] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    # 컬럼명 매핑 (InfluxDB → ColumnConfig)
    column_mapping = {
        "BR1_EO_FG_A": cc.col_o2,
        "SNR_PMP_UW_S_1": cc.col_hz,
        "ICF_SCS_FG_T_1": cc.col_outer_temp,
        "ICF_TMS_NOX_A": cc.col_nox,
        "ACT_STATUS": cc.col_act_status,
    }

    for influx_col, config_col in column_mapping.items():
        if influx_col in df.columns:
            df_mapped[config_col] = df[influx_col]

    # 필요한 컬럼만 추출
    required_cols = [cc.col_datetime] + list(column_mapping.values())
    df_mapped = df_mapped[required_cols].dropna(subset=[cc.col_datetime])

    print(f"🔄 컬럼 매핑 완료: {df_mapped.shape}")
    print(f"📋 매핑된 컬럼: {list(df_mapped.columns)}")

    # 2) 5초 윈도우 요약 (센서: 평균, 상태: 마지막값)
    df_mapped = df_mapped.set_index(cc.col_datetime).sort_index()

    # 센서/상태 컬럼 구분
    status_cols = [cc.col_act_status]
    sensor_cols = [c for c in df_mapped.columns if c not in status_cols]

    # 5초 윈도우 요약
    df_mean = (
        df_mapped[sensor_cols].resample("5s", label="right", closed="right").mean()
    )
    df_last = (
        df_mapped[status_cols].resample("5s", label="right", closed="right").last()
    )

    # 보간 전 요약 출력
    agg_pre = pd.concat([df_mean, df_last], axis=1)
    agg_pre.index.name = cc.col_datetime
    agg_pre = agg_pre.reset_index()
    agg_pre = agg_pre.sort_values(cc.col_datetime).head(120)
    print("🧾 5초 윈도우 요약(보간 전, UTC):")
    print(f"   📊 전체 행 수: {len(agg_pre)}")
    print(
        f"   📅 시간 범위: {agg_pre[cc.col_datetime].min()} ~ {agg_pre[cc.col_datetime].max()}"
    )
    print(agg_pre.head(4))
    print("🧾 5초 윈도우 요약(보간 전, 마지막 4개):")
    print(agg_pre.tail(4))

    # 3) preprocessor.py의 make_infer_ffill 활용
    print("🔧 preprocessor.py make_infer_ffill 적용 중...")
    agg_processed = preprocessor.make_infer_ffill(
        agg_pre,
        require_full_index=False,  # 이미 5초 간격으로 요약됨
        logger_cfg=LoggerConfig(name="MLflowInference", level=20),  # INFO 레벨
    )

    # 4) 최종 결과 정리
    agg_processed = agg_processed.sort_values(cc.col_datetime).head(120)
    print("🔧 make_infer_ffill 적용 후:")
    print(f"   📊 전체 행 수: {len(agg_processed)}")
    print(
        f"   📅 시간 범위: {agg_processed[cc.col_datetime].min()} ~ {agg_processed[cc.col_datetime].max()}"
    )
    print("🔧 make_infer_ffill 적용 후 (마지막 4개):")
    print(agg_processed.tail(4))

    # 컬럼명을 원래 REQUIRED_COLUMNS로 되돌리기
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    reverse_mapping[cc.col_datetime] = "_time_gateway"

    agg_final = agg_processed.rename(columns=reverse_mapping)

    # 열 순서 정렬: REQUIRED_COLUMNS 순서 유지(존재하는 것만)
    required_columns = [
        "_time_gateway",
        "BR1_EO_FG_A",
        "SNR_PMP_UW_S_1",
        "ICF_SCS_FG_T_1",
        "ICF_TMS_NOX_A",
        "ACT_STATUS",
    ]
    ordered_cols = [c for c in required_columns if c in agg_final.columns]
    agg_final = agg_final[ordered_cols]

    print("🧾 5초 윈도우 요약(보간 후, UTC):")
    print(f"   📊 최종 행 수: {len(agg_final)}")
    print(
        f"   📅 최종 시간 범위: {agg_final['_time_gateway'].min()} ~ {agg_final['_time_gateway'].max()}"
    )
    print(agg_final.head(4))

    return agg_final


def setup_preprocessing_config() -> tuple[
    ColumnConfig,
    InferPreprocessingConfig,
    LGBMInferPreprocessingConfig,
    Preprocessor,
    LGBMFeaturePreprocessor,
    GPModelConfig,
    LGBMModelConfig,
    GaussianProcessNOxModel,
    LGBMNOxModel,
    OptimizationConfig,
    PumpOptimizer,
    LGBMPumpHzAdjuster,
]:
    """SRDD용 전처리 설정 및 GP/LGBM 모델, PumpOptimizer 초기화"""
    # ColumnConfig 초기화 (SRDD 프리셋 적용)
    cc = ColumnConfig(plant_code="SRDD")

    # InferPreprocessingConfig 초기화
    infer_cfg = InferPreprocessingConfig(
        column_config=cc,
        plant_code="SRDD",
        resample_sec=5,  # 5초 간격
        ffill_limit_sec=600,  # 10분 이내 ffill
    )

    # LGBMInferPreprocessingConfig 초기화
    lgbm_infer_cfg = LGBMInferPreprocessingConfig(column_config=cc)

    # Preprocessor 초기화
    preprocessor = Preprocessor(
        column_config=cc,
        prep_infer_cfg=infer_cfg,
    )

    # LGBMFeaturePreprocessor 초기화
    lgbm_preprocessor = LGBMFeaturePreprocessor(lgbm_infer_cfg)

    # GPModelConfig 초기화
    gp_cfg = GPModelConfig(
        column_config=cc,
        plant_code="SRDD",
        logger_cfg=LoggerConfig(name="GPModel", level=20),  # INFO 레벨
    )

    # LGBMModelConfig 초기화
    lgbm_cfg = LGBMModelConfig(
        lgbm_feature_columns_original=cc.lgbm_feature_columns,
        lgbm_feature_columns_summary=[],  # 나중에 업데이트
        model_path="mlflow_artifacts/8df2907f144a4dcd80fe0d834be77f65/urea_gp_model/lgbm_model.joblib",
        logger_cfg=LoggerConfig(name="LGBMModel", level=20),
    )

    # GaussianProcessNOxModel 초기화
    gp_model = GaussianProcessNOxModel(
        column_config=cc,
        model_config=gp_cfg,
    )

    # LGBMNOxModel 초기화
    lgbm_model = LGBMNOxModel(
        column_config=cc,
        model_config=lgbm_cfg,
    )

    # OptimizationConfig 초기화 (기본값 사용)
    opt_cfg = OptimizationConfig()

    # RuleConfig 초기화
    rule_cfg = RuleConfig()

    # PumpOptimizer 초기화
    pump_optimizer = PumpOptimizer(
        model=gp_model,
        column_config=cc,
        opt_config=opt_cfg,
        rule_config=rule_cfg,
    )

    # LGBMPumpHzAdjuster 초기화
    lgbm_adjuster = LGBMPumpHzAdjuster(
        column_config=cc,
        model_config=lgbm_cfg,
        rule_config=rule_cfg,
        optimization_config=opt_cfg,
    )

    return (
        cc,
        infer_cfg,
        lgbm_infer_cfg,
        preprocessor,
        lgbm_preprocessor,
        gp_cfg,
        lgbm_cfg,
        gp_model,
        lgbm_model,
        opt_cfg,
        pump_optimizer,
        lgbm_adjuster,
    )


def query_recent_influx() -> pd.DataFrame:
    """SRDD InfluxDB에서 최근 데이터 조회"""
    from influxdb import InfluxDBClient

    host = os.environ.get("INFLUX_HOST", "10.238.24.150")
    port = int(os.environ.get("INFLUX_PORT", "8086"))
    username = os.environ.get("INFLUX_USERNAME", "read_user")
    password = os.environ.get("INFLUX_PASSWORD", "!Skepinfluxuser25")
    database = os.environ.get("INFLUX_DB", "SRDD")
    measurement = os.environ.get("INFLUX_MEASUREMENT", "SRDD")
    # 요구사항: 최근 10분 조회 (5초 간격 → 120개) 또는 절대 시작시각 기반 조회
    window = os.environ.get("INFLUX_WINDOW", "10m")
    limit = int(os.environ.get("INFLUX_LIMIT", "600"))
    start_time_kst = os.environ.get("START_TIME_KST", "").strip()
    start_time_utc = os.environ.get("START_TIME", "").strip()

    client = InfluxDBClient(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        timeout=30,
    )

    # 절대 시작시각이 지정되면 해당 구간만 조회 (우선순위: START_TIME(UTC) > START_TIME_KST)
    if start_time_utc:
        # UTC 기준 고정 구간
        start_utc_dt = pd.to_datetime(start_time_utc, utc=True, errors="coerce")
        # INFLUX_WINDOW 파싱 (s/m)
        w = window.lower().strip()
        secs = 600  # 10분 기본값
        if w.endswith("s"):
            secs = int(w[:-1] or 0)
        elif w.endswith("m"):
            secs = int(w[:-1] or 0) * 60
        else:
            # fallback: 10m
            secs = 600
        end_utc_dt = start_utc_dt + pd.to_timedelta(max(secs - 1, 0), unit="s")
        start_utc = start_utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc = end_utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(
            f"[INFO] 절대 시간 조회(UTC): {start_utc_dt} ~ {end_utc_dt} (window={window})"
        )
        query = (
            f'\nSELECT * FROM "{measurement}" '
            f"WHERE time >= '{start_utc}' AND time <= '{end_utc}' "
            f"ORDER BY time ASC LIMIT {limit}\n"
        )
    elif start_time_kst:
        try:
            start_kst = pd.to_datetime(start_time_kst).tz_localize("Asia/Seoul")
        except Exception:
            start_kst = pd.to_datetime(start_time_kst).tz_convert("Asia/Seoul")
        # INFLUX_WINDOW 파싱 (s/m)
        w = window.lower().strip()
        secs = 600  # 10분 기본값
        if w.endswith("s"):
            secs = int(w[:-1] or 0)
        elif w.endswith("m"):
            secs = int(w[:-1] or 0) * 60
        else:
            # fallback: 10m
            secs = 600
        # 종료 시점 포함 조건(<=)이므로 정확히 10분 구간을 만들기 위해 1초 감소
        end_kst = start_kst + pd.to_timedelta(max(secs - 1, 0), unit="s")
        start_utc = start_kst.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        end_utc = end_kst.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[INFO] 절대 시간 조회(KST): {start_kst} ~ {end_kst} (window={window})")
        query = (
            f'\nSELECT * FROM "{measurement}" '
            f"WHERE time >= '{start_utc}' AND time <= '{end_utc}' "
            f"ORDER BY time ASC LIMIT {limit}\n"
        )
    else:
        query = (
            f'\nSELECT * FROM "{measurement}" '
            f"WHERE time >= now() - {window} AND time <= now() "
            f"ORDER BY time DESC LIMIT {limit}\n"
        )
    print("🔎 Influx 쿼리:", query)
    result = client.query(query)
    points = list(result.get_points()) if result else []
    print(f"📊 조회 포인트 수: {len(points)}")

    if not points:
        raise RuntimeError("최근 구간 데이터가 없습니다. 시간창/측정값을 조정하세요.")

    df = pd.DataFrame(points)
    print("🗂️ 원본 데이터프레임:", df.shape)
    # 주 관심 컬럼만 미리보기
    preview_cols = [
        c
        for c in [
            "time",
            "BR1_EO_FG_A",
            "SNR_PMP_UW_S_1",
            "ICF_SCS_FG_T_1",
            "ICF_TMS_NOX_A",
            "ACT_STATUS",
        ]
        if c in df.columns
    ]
    try:
        print("🔍 원본 InfluxDB 데이터 (처음 20개 행):")
        print(df[preview_cols].head(20) if preview_cols else df.head(20))
        print("\n🔍 원본 InfluxDB 데이터 (마지막 20개 행):")
        print(df[preview_cols].tail(20) if preview_cols else df.tail(20))
        print("\n🔍 원본 InfluxDB 데이터 통계:")
        if preview_cols:
            print(df[preview_cols].describe())
        else:
            print(df.describe())
    except Exception:
        print("🔍 원본 InfluxDB 데이터 (처리 실패):")
        print(df.head(5))
    return df


def main() -> None:
    print("🚀" + "=" * 58)
    print("🚀 SRDD GP 모델 기반 실시간 추론 및 Hz 추천 테스트 시작")
    print("🚀" + "=" * 58)

    # 0) 전처리 설정 및 GP/LGBM 모델, PumpOptimizer 초기화 (주석처리 - 모델 미준비)
    print("⚙️ 전처리 설정 초기화 중...")
    (
        cc,
        infer_cfg,
        lgbm_infer_cfg,
        preprocessor,
        lgbm_preprocessor,
        gp_cfg,
        lgbm_cfg,
        gp_model,
        lgbm_model,
        opt_cfg,
        pump_optimizer,
        lgbm_adjuster,
    ) = setup_preprocessing_config()
    print(f"✅ 전처리 설정 초기화 완료: {cc.plant_code}")
    # print(f"✅ GP 모델 초기화 완료: {gp_model.model_config.plant_code}")
    # print(f"ℹ️ LGBM 모델 초기화 완료: {lgbm_model.model_config.__class__.__name__} (비활성화)")
    # print(f"✅ PumpOptimizer 초기화 완료")
    # print(f"ℹ️ LGBM Adjuster 초기화 완료 (비활성화)")

    # tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    # if tracking_uri:
    #     print(f"🔗 MLFLOW_TRACKING_URI: {tracking_uri}")
    # else:
    #     print(
    #         "⚠️ MLFLOW_TRACKING_URI가 설정되지 않았습니다. mlflow 기본 설정을 사용합니다."
    #     )

    # # 1) RUN 선택 (주석처리 - 모델 미준비)
    # if tracking_uri:
    #     import mlflow
    #     mlflow.set_tracking_uri(tracking_uri)
    # run_id = os.environ.get("RUN_ID", "8df2907f144a4dcd80fe0d834be77f65")
    # print(f"🏷️ 사용 RUN_ID: {run_id}")

    # 2) GP 모델 로드 (주석처리 - 모델 미준비)
    # model_file = f"mlflow_artifacts/{run_id}/urea_gp_model/gp_model.joblib"
    # if not os.path.exists(model_file):
    #     # 대안 경로 시도
    #     model_file = f"mlflow_artifacts/{run_id}/gp_model.joblib"
    #     if not os.path.exists(model_file):
    #         raise FileNotFoundError(f"GP 모델 파일을 찾을 수 없습니다: {model_file}")

    # # GP 모델 로드
    # gp_model.load(model_file)
    # print(f"✅ GP 모델 로드 완료: {model_file}")

    # 3) LGBM 모델 로드 (주석처리 - 모델 미준비)
    # lgbm_model_path = os.environ.get(
    #     "LGBM_MODEL_PATH", f"mlflow_artifacts/{run_id}/urea_gp_model/lgbm_model.joblib"
    # )
    # if not os.path.exists(lgbm_model_path):
    #     raise FileNotFoundError(f"LGBM 모델 파일을 찾을 수 없습니다: {lgbm_model_path}")

    # # LGBM 모델 로드
    # lgbm_model.load(lgbm_model_path)
    # print(f"✅ LGBM 모델 로드 완료: {lgbm_model_path}")

    # 4) Influx 최근 데이터 조회 (SRDD용)
    print("🔍 InfluxDB에서 실시간 데이터 조회 중...")
    df = query_recent_influx()

    if df.empty:
        print("❌ InfluxDB에서 데이터를 가져올 수 없습니다.")
        return

    print(f"📈 원본 데이터: {len(df)}행")
    print(f"📅 시간 범위: {df['time'].min()} ~ {df['time'].max()}")

    # 원본 데이터 시간 간격 검증
    print("\n🔍 원본 데이터 시간 간격 검증:")
    df_sorted = df.sort_values("time")
    # time 컬럼을 datetime으로 변환
    df_sorted["time_dt"] = pd.to_datetime(df_sorted["time"], utc=True)
    time_diffs = df_sorted["time_dt"].diff().dropna()
    print(f"   - 평균 시간 간격: {time_diffs.mean()}")
    print(f"   - 최소 시간 간격: {time_diffs.min()}")
    print(f"   - 최대 시간 간격: {time_diffs.max()}")
    print(f"   - 시간 간격 분포: {time_diffs.value_counts().head()}")

    # 5) 5초 윈도우 요약(최근 10분 → 120행) - SRS1과 동일한 로직 사용
    print("\n🔄 5초 윈도우 요약 처리 중...")
    agg = aggregate_10min_to_5s(df, preprocessor, cc)
    print("🧾 5초 윈도우 요약 완료:", agg.shape)

    # 5초 윈도우 검증
    print("\n🔍 5초 윈도우 검증:")
    agg_sorted = agg.sort_values("_time_gateway")
    window_diffs = agg_sorted["_time_gateway"].diff().dropna()
    print(f"   - 예상 행 수: 120개 (10분 ÷ 5초)")
    print(f"   - 실제 행 수: {len(agg)}개")
    print(f"   - 평균 윈도우 간격: {window_diffs.mean()}")
    print(
        f"   - 윈도우 간격이 5초인지 확인: {all(window_diffs == pd.Timedelta(seconds=5))}"
    )

    print("\n📊 요약된 데이터 (처음 10개 행):")
    print(agg.head(10))
    print("\n📊 요약된 데이터 (마지막 10개 행):")
    print(agg.tail(10))

    # 시간 범위 검증
    print(f"\n📅 시간 범위 검증:")
    print(f"   - 시작: {agg['_time_gateway'].min()}")
    print(f"   - 끝: {agg['_time_gateway'].max()}")
    print(
        f"   - 총 기간: {(agg['_time_gateway'].max() - agg['_time_gateway'].min()).total_seconds()}초"
    )
    print(f"   - 예상 기간: 595초 (10분 - 5초)")

    # 6) 데이터 품질 확인
    print("\n📊 데이터 품질 확인:")
    print(f"   - 전체 행 수: {len(agg)}")
    print(f"   - 컬럼 수: {len(agg.columns)}")
    print(f"   - 컬럼 목록: {list(agg.columns)}")

    # 주요 센서 데이터 확인
    sensor_cols = ["BR1_EO_FG_A", "SNR_PMP_UW_S_1", "ICF_SCS_FG_T_1", "ICF_TMS_NOX_A"]
    for col in sensor_cols:
        if col in agg.columns:
            non_null_count = agg[col].count()
            null_count = agg[col].isnull().sum()
            print(f"   - {col}: {non_null_count}개 유효값, {null_count}개 NULL")
        else:
            print(f"   - {col}: 컬럼 없음")

    # 결측치가 있는 행 확인
    missing_data = agg.isnull().any(axis=1).sum()
    print(f"   - 결측치가 있는 행: {missing_data}개")

    if missing_data > 0:
        print("   - 결측치가 있는 행들:")
        missing_rows = agg[agg.isnull().any(axis=1)]
        for idx, row in missing_rows.head(3).iterrows():
            print(
                f"     행 {idx}: {row['_time_gateway']} - {row.isnull().sum()}개 결측치"
            )

    print("\n📌 요약")
    print("- 플랜트 코드:", cc.plant_code)
    print("- 원본 데이터 행 수:", len(df))
    print("- 5초 윈도우 요약 행 수:", len(agg))
    print("- 데이터 품질 확인 완료")
    print("- 모델 추론: 주석처리됨 (모델 미준비)")
    print("✅ SRDD 실시간 데이터 조회 및 5초 윈도우 요약 테스트 완료!")


if __name__ == "__main__":
    main()
