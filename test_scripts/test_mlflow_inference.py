"""
test_mlflow_inference.py

기능: (1) RUN_ID로 모델 다운로드(또는 최신 RUN 자동 선택),
     (2) Influx에서 최근 데이터 조회, (3) 간단 전처리,
     (4) 모델 로드/추론, (5) 콘솔에 결과/중간값 출력

환경변수(권장):
  - MLFLOW_TRACKING_URI: 예) http://10.250.109.206:5000
  - RUN_ID: 지정 시 해당 RUN 사용, 미지정 시 최신 RUN 자동 선택
  - MLFLOW_EXPERIMENT_NAME: 최신 RUN 자동 선택 시 필요 (예: urea_gp_prod)

  - MODEL_LOCAL_DIR: 수동 복사한 모델 디렉토리 지정 시 MLflow 다운로드 우회
    (미지정 시, 프로젝트 내 mlflow_artifacts/<RUN_ID>/urea_gp_model 경로 자동 탐색)

  - START_TIME (UTC): 절대 시작시각(UTC) 지정 시 사용 (예: "2025-08-27 00:00:01")
    - 지정 시: [START_TIME, START_TIME + INFLUX_WINDOW] 구간만 조회 (UTC)
  - START_TIME_KST: 절대 시작시각(KST) 지정 시 사용 (예: "2025-08-27 09:00:01")
    - 지정 시: [START_TIME_KST, START_TIME_KST + INFLUX_WINDOW] 구간만 조회 (KST)
    - 미지정 시: [now() - INFLUX_WINDOW, now()] 구간 조회

  - INFLUX_HOST (기본: 10.238.27.132)
  - INFLUX_PORT (기본: 8086)
  - INFLUX_USERNAME (기본: read_user)
  - INFLUX_PASSWORD (기본: !Skepinfluxuser25)
  - INFLUX_DB (기본: SRS1)
  - INFLUX_MEASUREMENT (기본: SRS1)
  - INFLUX_WINDOW (기본: 10m)
  - INFLUX_LIMIT (기본: 120)
"""

import os
import sys
from pathlib import Path
from typing import Optional, List

# Ensure project root is on sys.path regardless of current working directory
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib

from influxdb import InfluxDBClient
import mlflow
import time
import requests

# 새로운 전처리 파이프라인 import
from config.column_config import ColumnConfig
from config.preprocessing_config import InferPreprocessingConfig
from config.model_config import GPModelConfig
from src.data_processing.preprocessor import Preprocessor
from src.models.gaussian_process import GaussianProcessNOxModel
from utils.logger import LoggerConfig

# pandas 출력 설정: 모든 컬럼 표시
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# GP 제어 모델 입력 요구 컬럼(8개) - ColumnConfig와 매핑
REQUIRED_COLUMNS: List[str] = [
    "_time_gateway",  # col_datetime
    "BR1_EO_O2_A",  # col_o2
    "SNR_PMP_UW_S_1",  # col_hz
    "ICF_CCS_FG_T_1",  # col_inner_temp
    "ICF_SCS_FG_T_1",  # col_outer_temp
    "ICF_TMS_NOX_A",  # col_nox
    "ACC_SNR_AI_1A",  # col_ai
    "ACT_STATUS",  # col_act_status
]


def get_env(name: str, default: Optional[str] = None) -> str:
    v = os.environ.get(name)
    return v if v is not None else ("" if default is None else str(default))


def setup_preprocessing_config() -> tuple[
    ColumnConfig,
    InferPreprocessingConfig,
    Preprocessor,
    GPModelConfig,
    GaussianProcessNOxModel,
]:
    """전처리 설정 및 GP 모델 초기화"""
    # ColumnConfig 초기화 (SRS1 프리셋 적용)
    cc = ColumnConfig(plant_code="SRS1")

    # InferPreprocessingConfig 초기화
    infer_cfg = InferPreprocessingConfig(
        column_config=cc,
        plant_code="SRS1",
        resample_sec=5,  # 5초 간격
        ffill_limit_sec=20,  # 20초 이내 ffill
    )

    # Preprocessor 초기화
    preprocessor = Preprocessor(
        column_config=cc,
        prep_infer_cfg=infer_cfg,
    )

    # GPModelConfig 초기화
    gp_cfg = GPModelConfig(
        column_config=cc,
        plant_code="SRS1",
        logger_cfg=LoggerConfig(name="GPModel", level=20),  # INFO 레벨
    )

    # GaussianProcessNOxModel 초기화
    gp_model = GaussianProcessNOxModel(
        column_config=cc,
        model_config=gp_cfg,
    )

    return cc, infer_cfg, preprocessor, gp_cfg, gp_model


def select_run_id() -> str:
    run_id = os.environ.get("RUN_ID")
    if run_id:
        print(f"[INFO] 환경변수 RUN_ID 지정됨: {run_id}")
        return run_id

    # 기본 실험명과 Run Name 접두어
    experiment = os.environ.get("MLFLOW_EXPERIMENT_NAME", "skep-urea")
    run_name_prefix = os.environ.get("MLFLOW_RUN_NAME_PREFIX", "urea-SRS1-")

    # Run Name으로 필터링하여 최신 RUN 선택
    filter_string = f"tags.mlflow.runName LIKE '{run_name_prefix}%'"

    print(f"[INFO] 최신 RUN 자동 선택 - 실험명: {experiment}")
    print(f"[INFO] 필터: {filter_string}")
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        raise ValueError(f"Experiment '{experiment}' not found")
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(
            f"No runs found in experiment '{experiment}' with filter '{filter_string}'"
        )
    return runs.iloc[0]["run_id"]


def download_model(run_id: str, model_name: str = "urea_gp_model") -> Path:
    # 0) 로컬 경로 오버라이드(문제시 수동 배포 파일 사용)
    override = os.environ.get("MODEL_LOCAL_DIR")
    if override:
        p = Path(override)
        if p.exists() and any(p.rglob("*")):
            print(f"[INFO] 로컬 모델 경로 사용(MODEL_LOCAL_DIR): {p}")
            return p

    # 0-1) 프로젝트 내 수동 복사본 자동 탐색
    # 우선순위: mlflow_artifacts/<RUN_ID>/urea_gp_model → mlflow_artifacts/<RUN_ID>/artifacts/urea_gp_model → mlflow_artifacts/<RUN_ID>
    local_candidates = [
        PROJECT_ROOT / "mlflow_artifacts" / run_id / model_name,
        PROJECT_ROOT / "mlflow_artifacts" / run_id / "artifacts" / model_name,
        PROJECT_ROOT / "mlflow_artifacts" / run_id,
    ]
    for cand in local_candidates:
        if cand.exists() and any(cand.rglob("*")):
            print(f"[INFO] 로컬 모델 경로 자동 감지: {cand}")
            return cand

    dst = Path("/tmp/mlflow_models") / f"{run_id}_{model_name}"
    # 캐시 존재 시 재사용
    if dst.exists() and any(dst.rglob("*")):
        print(f"[INFO] 캐시된 모델 사용: {dst}")
        return dst

    print(f"[INFO] 모델 다운로드 시작: run_id={run_id}, artifact={model_name}")
    t0 = time.time()
    path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/{model_name}",
        dst_path=str(dst),
    )
    elapsed = time.time() - t0
    print(f"[INFO] 모델 다운로드 완료 ({elapsed:.1f}s): {path}")
    print("[INFO] 포함 파일 목록(최대 10개):")
    cnt = 0
    for p in Path(path).rglob("*"):
        print(" -", p)
        cnt += 1
        if cnt >= 10:
            print(" - ...")
            break
    return path


def test_mlflow_connection() -> None:
    """Quick MLflow connectivity test with short timeouts.

    - GET tracking root (5s)
    - POST experiments/list (5s)
    - If RUN_ID is set, POST runs/get to print artifact_uri (5s)
    """
    tracking = os.environ.get("MLFLOW_TRACKING_URI")
    print("\n🧪 MLflow 연결 테스트")
    if not tracking:
        print("[WARN] MLFLOW_TRACKING_URI 미설정 → 연결 테스트 건너뜀")
        return

    base = tracking.rstrip("/")
    try:
        r = requests.get(base, timeout=5)
        print(f"  ↳ GET {base} → HTTP {r.status_code}")
    except Exception as e:
        print(f"❌ GET {base} 실패: {e}")

    try:
        url = f"{base}/api/2.0/mlflow/experiments/list"
        r = requests.post(url, json={}, timeout=5)
        print(f"  ↳ POST /experiments/list → HTTP {r.status_code}")
    except Exception as e:
        print(f"❌ POST /experiments/list 실패: {e}")

    run_id = os.environ.get("RUN_ID")
    if run_id:
        try:
            url = f"{base}/api/2.0/mlflow/runs/get"
            r = requests.post(url, json={"run_id": run_id}, timeout=5)
            if r.ok:
                data = r.json()
                art = data.get("run", {}).get("info", {}).get("artifact_uri")
                print(f"📦 run.artifact_uri: {art}")
            else:
                print(f"⚠️ runs/get HTTP {r.status_code}")
        except Exception as e:
            print(f"❌ POST /runs/get 실패: {e}")
    else:
        print("ℹ️ RUN_ID 미설정 → runs/get 생략")


def pick_model_file(root: Path) -> Path:
    candidates: List[Path] = [*root.rglob("*.joblib"), *root.rglob("*.pkl")]
    if not candidates:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {root}")
    print("[INFO] 로드 후보:")
    for c in candidates:
        print(" -", c)
    return candidates[0]


def query_recent_influx() -> pd.DataFrame:
    host = get_env("INFLUX_HOST", "10.238.27.132")
    port = int(get_env("INFLUX_PORT", "8086"))
    username = get_env("INFLUX_USERNAME", "read_user")
    password = get_env("INFLUX_PASSWORD", "!Skepinfluxuser25")
    database = get_env("INFLUX_DB", "SRS1")
    measurement = get_env("INFLUX_MEASUREMENT", "SRS1")
    # 요구사항: 최근 20초 조회 (초당 1포인트 가정 → 20개) 또는 절대 시작시각 기반 조회
    window = get_env("INFLUX_WINDOW", "20s")
    limit = int(get_env("INFLUX_LIMIT", "200"))
    start_time_kst = get_env("START_TIME_KST", "").strip()
    start_time_utc = get_env("START_TIME", "").strip()

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
        secs = 20
        if w.endswith("s"):
            secs = int(w[:-1] or 0)
        elif w.endswith("m"):
            secs = int(w[:-1] or 0) * 60
        else:
            # fallback: 20s
            secs = 20
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
        secs = 20
        if w.endswith("s"):
            secs = int(w[:-1] or 0)
        elif w.endswith("m"):
            secs = int(w[:-1] or 0) * 60
        else:
            # fallback: 20s
            secs = 20
        # 종료 시점 포함 조건(<=)이므로 정확히 20초 구간을 만들기 위해 1초 감소
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
    # 주 관심 컬럼만 미리보기: REQUIRED_COLUMNS + time
    preview_cols = [c for c in ["time", *REQUIRED_COLUMNS] if c in df.columns]
    try:
        print(df[preview_cols].head(20) if preview_cols else df.head(20))
    except Exception:
        print(df.head(20))
    return df


def aggregate_last_20s_to_5s(
    df: pd.DataFrame, preprocessor: Preprocessor, cc: ColumnConfig
) -> pd.DataFrame:
    """최근 20초 데이터를 5초 윈도우로 요약하여 4행 반환.

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
    agg_pre = agg_pre.sort_values(cc.col_datetime).head(4)
    print("🧾 5초 윈도우 요약(보간 전, UTC):")
    print(agg_pre.tail(4))

    # 3) preprocessor.py의 make_infer_ffill 활용
    print("🔧 preprocessor.py make_infer_ffill 적용 중...")
    agg_processed = preprocessor.make_infer_ffill(
        agg_pre,
        require_full_index=False,  # 이미 5초 간격으로 요약됨
        logger_cfg=LoggerConfig(name="MLflowInference", level=20),  # INFO 레벨
    )

    # 4) 최종 결과 정리
    agg_processed = agg_processed.sort_values(cc.col_datetime).head(4)

    # 컬럼명을 원래 REQUIRED_COLUMNS로 되돌리기
    reverse_mapping = {v: k for k, v in column_mapping.items()}
    reverse_mapping[cc.col_datetime] = "_time_gateway"

    agg_final = agg_processed.rename(columns=reverse_mapping)

    # 열 순서 정렬: REQUIRED_COLUMNS 순서 유지(존재하는 것만)
    ordered_cols = [c for c in REQUIRED_COLUMNS if c in agg_final.columns]
    agg_final = agg_final[ordered_cols]

    print("🧾 5초 윈도우 요약(보간 후, UTC):")
    print(agg_final.tail(4))

    return agg_final


def main() -> None:
    print("🚀" + "=" * 58)
    print("🚀 GP 모델 기반 실시간 추론 테스트 시작")
    print("🚀" + "=" * 58)

    # 0) 전처리 설정 및 GP 모델 초기화
    print("⚙️ 전처리 설정 및 GP 모델 초기화 중...")
    cc, infer_cfg, preprocessor, gp_cfg, gp_model = setup_preprocessing_config()
    print(f"✅ 전처리 설정 완료: {cc.plant_code}")
    print(f"✅ GP 모델 초기화 완료: {gp_model.model_config.plant_code}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"🔗 MLFLOW_TRACKING_URI: {tracking_uri}")
    else:
        print(
            "⚠️ MLFLOW_TRACKING_URI가 설정되지 않았습니다. mlflow 기본 설정을 사용합니다."
        )

    # 1) RUN 선택
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    # 연결 사전 점검
    test_mlflow_connection()
    run_id = select_run_id()
    print(f"🏷️ 사용 RUN_ID: {run_id}")

    # 2) GP 모델 로드 (기존 MLflow 모델 대신)
    model_file = f"mlflow_artifacts/{run_id}/urea_gp_model/gp_model.joblib"
    if not os.path.exists(model_file):
        # 대안 경로 시도
        model_file = f"mlflow_artifacts/{run_id}/gp_model.joblib"
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"GP 모델 파일을 찾을 수 없습니다: {model_file}")

    # GP 모델 로드
    gp_model.load(model_file)
    print(f"✅ GP 모델 로드 완료: {model_file}")

    # 모델 정보 출력
    model_info = gp_model.get_model_info()
    print(
        f"📊 모델 정보: {model_info['status']}, 학습 샘플: {model_info.get('n_train', 'N/A')}"
    )

    # 4) Influx 최근 데이터 조회
    df = query_recent_influx()

    # 5) 5초 윈도우 요약(최근 20초 → 4행) - 새로운 전처리 파이프라인 활용
    agg = aggregate_last_20s_to_5s(df, preprocessor, cc)
    print("🧾 모델 입력용 요약(열 순서 고정):", agg.shape)
    print(agg)

    # 6) 모델 입력행 만들기: ColumnConfig의 gp_feature_columns 활용
    feature_cols = cc.gp_feature_columns  # [col_hz, col_o2, col_temp]
    # InfluxDB 컬럼명으로 변환
    influx_feature_cols = ["SNR_PMP_UW_S_1", "BR1_EO_O2_A", "ICF_CCS_FG_T_1"]

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
    print(X)

    # 7) GP 모델 예측: 5초 윈도우 평균 입력만 사용하여 각 시점의 NOx 예측 (결측 구간 제외)
    if len(X) > 0:
        print("🧠 GP 모델 예측 시작...")
        pred_mean, pred_std = gp_model.predict(X, return_std=True)

        # 예측 결과를 DataFrame에 추가
        agg_with_pred = agg.copy()
        agg_with_pred[cc.col_pred_mean] = np.nan
        agg_with_pred[cc.col_pred_ucb] = np.nan

        # 유효한 예측만 결과에 반영
        for i, (t, mean_val, std_val) in enumerate(
            zip(valid_times, pred_mean, pred_std)
        ):
            # UCB (Upper Confidence Bound) 계산
            ucb_val = mean_val + 1.96 * std_val  # 95% 신뢰구간 상한

            # 결과 출력
            print(
                f"🎯 {t} → NOx mean={float(mean_val):.3f} ± {float(std_val):.3f} (UCB: {float(ucb_val):.3f})"
            )

            # DataFrame에 결과 저장
            mask = agg_with_pred["_time_gateway"] == t
            agg_with_pred.loc[mask, cc.col_pred_mean] = mean_val
            agg_with_pred.loc[mask, cc.col_pred_ucb] = ucb_val

        # 최종 결과 출력
        print("\n📊 최종 예측 결과:")
        result_cols = ["_time_gateway", cc.col_pred_mean, cc.col_pred_ucb]
        print(agg_with_pred[result_cols].dropna())

    else:
        print("⚠️ 예측 가능한(결측 없는) 5초 구간이 없습니다.")

    # 결측으로 제외된 구간은 NaN으로 표시
    for t in invalid_times:
        print(f"⚪ {t} → NOx mean=NaN (insufficient data)")

    print("\n📌 요약")
    print("- RUN_ID:", run_id)
    print("- 모델 경로:", model_file)
    print("- 입력 요약 행 수:", len(agg))


if __name__ == "__main__":
    main()
