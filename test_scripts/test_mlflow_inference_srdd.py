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
        "BR1_EO_O2_A": cc.col_o2,
        "SNR_PMP_UW_S_1": cc.col_hz,
        "ICF_CCS_FG_T_1": cc.col_inner_temp,
        "ICF_SCS_FG_T_1": cc.col_outer_temp,
        "ICF_TMS_NOX_A": cc.col_nox,
        "ACC_SNR_AI_1A": cc.col_ai,
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
        "BR1_EO_O2_A",
        "SNR_PMP_UW_S_1",
        "ICF_CCS_FG_T_1",
        "ICF_SCS_FG_T_1",
        "ICF_TMS_NOX_A",
        "ACC_SNR_AI_1A",
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
            "BR1_EO_O2_A",
            "SNR_PMP_UW_S_1",
            "ICF_CCS_FG_T_1",
            "ICF_SCS_FG_T_1",
            "ICF_TMS_NOX_A",
            "ACC_SNR_AI_1A",
            "ACT_STATUS",
        ]
        if c in df.columns
    ]
    try:
        print("🔍 원본 InfluxDB 데이터 (처음 5개 행):")
        print(df[preview_cols].head(5) if preview_cols else df.head(5))
        print("🔍 원본 InfluxDB 데이터 통계:")
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

    # 0) 전처리 설정 및 GP/LGBM 모델, PumpOptimizer 초기화
    print("⚙️ 전처리 설정 및 GP/LGBM 모델, PumpOptimizer 초기화 중...")
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
    print(f"✅ GP 모델 초기화 완료: {gp_model.model_config.plant_code}")
    print(
        f"ℹ️ LGBM 모델 초기화 완료: {lgbm_model.model_config.__class__.__name__} (비활성화)"
    )
    print(f"✅ PumpOptimizer 초기화 완료")
    print(f"ℹ️ LGBM Adjuster 초기화 완료 (비활성화)")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"🔗 MLFLOW_TRACKING_URI: {tracking_uri}")
    else:
        print(
            "⚠️ MLFLOW_TRACKING_URI가 설정되지 않았습니다. mlflow 기본 설정을 사용합니다."
        )

    # 1) RUN 선택
    if tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
    run_id = os.environ.get("RUN_ID", "8df2907f144a4dcd80fe0d834be77f65")
    print(f"🏷️ 사용 RUN_ID: {run_id}")

    # 2) GP 모델 로드
    model_file = f"mlflow_artifacts/{run_id}/urea_gp_model/gp_model.joblib"
    if not os.path.exists(model_file):
        # 대안 경로 시도
        model_file = f"mlflow_artifacts/{run_id}/gp_model.joblib"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"GP 모델 파일을 찾을 수 없습니다: {model_file}")

    # GP 모델 로드
    gp_model.load(model_file)
    print(f"✅ GP 모델 로드 완료: {model_file}")

    # 3) LGBM 모델 로드
    lgbm_model_path = os.environ.get(
        "LGBM_MODEL_PATH", f"mlflow_artifacts/{run_id}/urea_gp_model/lgbm_model.joblib"
    )
    if not os.path.exists(lgbm_model_path):
        raise FileNotFoundError(f"LGBM 모델 파일을 찾을 수 없습니다: {lgbm_model_path}")

    # LGBM 모델 로드
    lgbm_model.load(lgbm_model_path)
    print(f"✅ LGBM 모델 로드 완료: {lgbm_model_path}")

    # 4) Influx 최근 데이터 조회 (SRDD용)
    df = query_recent_influx()

    if df.empty:
        print("❌ InfluxDB에서 데이터를 가져올 수 없습니다.")
        return

    print(f"📈 원본 데이터: {len(df)}행")
    print(f"📅 시간 범위: {df['_time_gateway'].min()} ~ {df['_time_gateway'].max()}")

    # 5) 5초 윈도우 요약(최근 10분 → 120행) - SRS1과 동일한 로직 사용
    agg = aggregate_10min_to_5s(df, preprocessor, cc)
    print("🧾 모델 입력용 요약(열 순서 고정):", agg.shape)
    print(agg)

    # 6) 모델 입력행 만들기: ColumnConfig의 gp_feature_columns 활용
    feature_cols = cc.gp_feature_columns  # SRDD 프리셋에 따라 col_outer_temp 사용
    # InfluxDB 컬럼명으로 변환 (SRDD용)
    influx_feature_cols = [
        "SNR_PMP_UW_S_1",
        "BR1_EO_O2_A",
        "ICF_SCS_FG_T_1",
    ]  # SRDD는 출구온도 사용

    missing_feat = [c for c in influx_feature_cols if c not in agg.columns]
    if missing_feat:
        raise KeyError(f"모델 입력 피처 누락: {missing_feat}")

    X_all = agg[influx_feature_cols]
    valid_mask = ~X_all.isna().any(axis=1)
    invalid_times = agg.loc[~valid_mask, "_time_gateway"].tolist()
    if invalid_times:
        print(
            f"[WARN] 결측치로 인해 예측에서 제외된 5초 구간: {len(invalid_times)}건 → {invalid_times}"
        )

    X = X_all.loc[valid_mask].to_numpy(dtype=float)
    valid_times = agg.loc[valid_mask, "_time_gateway"].tolist()
    print("🧮 예측 입력 배열 형태:", X.shape)
    print(f"📋 피처 컬럼: {influx_feature_cols}")
    print("🔍 GP 모델 입력 데이터 (처음 5개 행):")
    print(X[:5])

    # 7) GP 모델 예측 및 Hz 추천: 각 5초 윈도우에 대해 NOx 예측 및 Hz 추천
    if len(X) > 0:
        print("🧠 GP 모델 예측 및 Hz 추천 시작...")

        # 예측 결과를 저장할 DataFrame 준비
        agg_with_recommendations = agg.copy()

        # GP 모델 일괄 예측 (120개 행)
        print("📊 GP 모델 일괄 예측 중...")
        gp_pred_mean, gp_pred_std = gp_model.predict(X, return_std=True)
        gp_pred_ucb = gp_pred_mean + 1.96 * gp_pred_std  # 95% 신뢰구간 상한

        # 각 유효한 시점에 대해 Hz 추천 수행
        for i, (t, x_row) in enumerate(zip(valid_times, X)):
            if i < 5:  # 처음 5개만 상세 출력
                print(f"\n🎯 시점 {i+1}: {t}")
                print(
                    f"   📊 NOx 예측: mean={gp_pred_mean[i]:.3f} ± {gp_pred_std[i]:.3f} (UCB: {gp_pred_ucb[i]:.3f})"
                )

            # PumpOptimizer를 위한 입력 데이터 준비
            current_row = agg[agg["_time_gateway"] == t].iloc[0]

            # Hz 추천 수행
            try:
                recommendation = pump_optimizer.predict_pump_hz(
                    target_nox=opt_cfg.target_nox,
                    pump_bounds=opt_cfg.pump_bounds,
                    current_oxygen=float(current_row["BR1_EO_O2_A"]),
                    current_temp=float(
                        current_row["ICF_SCS_FG_T_1"]
                    ),  # SRDD는 출구온도 사용
                    current_target=float(current_row["ICF_TMS_NOX_A"]),
                    p_feasible=opt_cfg.p_feasible,
                    n_candidates=opt_cfg.n_candidates,
                    round_to_int=opt_cfg.round_to_int,
                )

                # DataFrame에 결과 저장
                mask = agg_with_recommendations["_time_gateway"] == t
                agg_with_recommendations.loc[mask, cc.col_pred_mean] = recommendation[
                    cc.col_pred_mean
                ]
                agg_with_recommendations.loc[mask, cc.col_pred_ucb] = recommendation[
                    cc.col_pred_ucb
                ]
                agg_with_recommendations.loc[mask, cc.col_hz_out] = recommendation[
                    cc.col_hz_out
                ]
                agg_with_recommendations.loc[mask, cc.col_safety_gap] = recommendation[
                    cc.col_safety_gap
                ]

                # PumpOptimizer의 규칙 후처리 적용
                df_single = pd.DataFrame(
                    [
                        {
                            cc.col_datetime: t,
                            cc.col_o2: float(current_row["BR1_EO_O2_A"]),
                            cc.col_temp: float(
                                current_row["ICF_SCS_FG_T_1"]
                            ),  # SRDD는 출구온도
                            cc.col_inner_temp: float(current_row["ICF_CCS_FG_T_1"]),
                            cc.col_outer_temp: float(current_row["ICF_SCS_FG_T_1"]),
                            cc.col_nox: float(current_row["ICF_TMS_NOX_A"]),
                            cc.col_hz_raw_out: recommendation[cc.col_hz_out],
                        }
                    ]
                )

                # 규칙 후처리 적용
                df_with_rules = pump_optimizer.add_rule_columns(df_single)

                # 4개 Hz 컬럼 모두 저장
                agg_with_recommendations.loc[mask, cc.col_hz_raw_out] = df_with_rules[
                    cc.col_hz_raw_out
                ].iloc[0]
                agg_with_recommendations.loc[mask, cc.col_hz_init_rule] = df_with_rules[
                    cc.col_hz_init_rule
                ].iloc[0]
                agg_with_recommendations.loc[mask, cc.col_hz_full_rule] = df_with_rules[
                    cc.col_hz_full_rule
                ].iloc[0]

                if i < 5:  # 처음 5개만 상세 출력
                    selected_hz = df_with_rules[cc.col_hz_raw_out].iloc[0]
                    selected_nox_mean = recommendation[cc.col_pred_mean]
                    selected_nox_ucb = recommendation[cc.col_pred_ucb]
                    safety_gap = recommendation[cc.col_safety_gap]
                    print(f"   🎛️ Hz 추천 (GP): {selected_hz:.1f} Hz")
                    print(
                        f"   📈 예측 NOx: {selected_nox_mean:.3f} (UCB: {selected_nox_ucb:.3f})"
                    )
                    print(f"   🛡️ 안전 여유: {safety_gap:.3f}")

            except Exception as e:
                if i < 5:  # 처음 5개만 상세 출력
                    print(f"   ❌ Hz 추천 실패: {e}")
                # fallback Hz 사용
                fallback_hz = 43.0
                mask = agg_with_recommendations["_time_gateway"] == t
                agg_with_recommendations.loc[mask, cc.col_hz_out] = fallback_hz

        print(f"\n✅ GP 모델 예측 완료: {len(valid_times)}개 시점")

        # 8) LGBM 모델 예측 및 Hz 조정
        print("\n🧠 LGBM 모델 예측 및 Hz 조정 시작...")

        # LGBM 전처리: 요약통계량 Feature 생성
        df_mapped = agg_with_recommendations.copy()
        column_mapping = {
            "BR1_EO_O2_A": "br1_eo_o2_a",
            "ICF_CCS_FG_T_1": "icf_ccs_fg_t_1",
            "ICF_SCS_FG_T_1": "icf_scs_fg_t_1",
            "ICF_TMS_NOX_A": "icf_tms_nox_a",
        }
        for influx_col, config_col in column_mapping.items():
            if influx_col in df_mapped.columns:
                df_mapped[config_col] = df_mapped[influx_col]

        # LGBM 전처리 (매핑된 DataFrame 사용)
        lgbm_suggested_df, lgbm_cols_x_stat = lgbm_preprocessor.make_interval_features(
            df_mapped
        )

        # LGBM 모델 설정 업데이트
        lgbm_cols_x_original = cc.lgbm_feature_columns
        lgbm_cfg.lgbm_feature_columns_original = list(lgbm_cols_x_original)
        lgbm_cfg.lgbm_feature_columns_summary = list(lgbm_cols_x_stat)
        lgbm_cfg.native_model_path = lgbm_model_path

        # LGBM 모델 예측 및 Hz 조정
        lgbm_suggested_df = lgbm_adjuster.predict_and_adjust(
            lgbm_suggested_df, return_flags=True
        )

        # LGBM 결과를 원본 DataFrame에 병합
        lgbm_result_cols = [cc.col_lgbm_db_pred_nox, cc.col_lgbm_db_hz_lgbm_adj]
        for col in lgbm_result_cols:
            if col in lgbm_suggested_df.columns:
                agg_with_recommendations[col] = lgbm_suggested_df[col].values

        # col_hz_final 설정 (LGBM 결과 사용)
        agg_with_recommendations[cc.col_hz_final] = agg_with_recommendations[
            cc.col_lgbm_db_hz_lgbm_adj
        ]

        print("✅ LGBM 모델 예측 및 Hz 조정 완료")

        # 최종 결과 출력 (처음 10개 행만)
        print("\n📊 최종 추천 결과 (처음 10개 행):")
        result_cols = [
            "_time_gateway",
            cc.col_pred_mean,  # pred_nox_mean
            cc.col_pred_ucb,  # pred_nox_ucb
            cc.col_hz_raw_out,  # act_snr_pmp_bo_1 (GP 결과)
            cc.col_hz_init_rule,  # act_snr_pmp_bo_2 (O2 규칙 적용)
            cc.col_hz_full_rule,  # act_snr_pmp_bo_3 (O2 + 동적 규칙)
            cc.col_safety_gap,  # safety_gap_to_target
        ]

        # LGBM 컬럼 추가
        if cc.col_lgbm_db_pred_nox in agg_with_recommendations.columns:
            result_cols.append(cc.col_lgbm_db_pred_nox)  # snr_nox_pred (LGBM 예측 NOx)
        if cc.col_lgbm_db_hz_lgbm_adj in agg_with_recommendations.columns:
            result_cols.append(
                cc.col_lgbm_db_hz_lgbm_adj
            )  # act_snr_pmp_bo_4 (LGBM 조정 Hz)

        # col_hz_final 추가 (최종 Hz 추천 값 - LGBM 반영된 결과)
        if cc.col_hz_final in agg_with_recommendations.columns:
            result_cols.append(cc.col_hz_final)  # act_snr_pmp_bo_0 (최종 추천 Hz)

        available_cols = [
            c for c in result_cols if c in agg_with_recommendations.columns
        ]
        print(agg_with_recommendations[available_cols].head(10))

    else:
        print("⚠️ 예측 가능한(결측 없는) 5초 구간이 없습니다.")

    # 결측으로 제외된 구간은 NaN으로 표시
    for t in invalid_times:
        print(f"⚪ {t} → NOx mean=NaN (insufficient data)")

    print("\n📌 요약")
    print("- RUN_ID:", run_id)
    print("- GP 모델 경로:", model_file)
    print("- LGBM 모델 경로:", lgbm_model_path)
    print("- 입력 요약 행 수:", len(agg))
    print("- GP 모델 예측 완료: PumpOptimizer 활용")
    print("- LGBM 모델 예측 완료: Hz 조정 반영")


if __name__ == "__main__":
    main()
