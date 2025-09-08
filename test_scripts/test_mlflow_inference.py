"""
test_mlflow_inference.py

기능: (1) RUN_ID로 모델 다운로드(또는 최신 RUN 자동 선택),
     (2) Influx에서 최근 데이터 조회, (3) 간단 전처리,
     (4) 모델 로드/추론, (5) 콘솔에 결과/중간값 출력

환경변수(권장):
  - MLFLOW_TRACKING_URI: 예) http://10.250.109.206:5000
  - RUN_ID: 지정 시 해당 RUN 사용, 미지정 시 최신 RUN 자동 선택
  - MLFLOW_EXPERIMENT_NAME: 최신 RUN 자동 선택 시 필요 (예: urea_gp_prod)

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

from repository.model import ModelRepository

# GP 제어 모델 입력 요구 컬럼(8개)
REQUIRED_COLUMNS: List[str] = [
    "_time_gateway",
    "BR1_EO_O2_A",
    "SNR_PMP_UW_S_1",
    "ICF_CCS_FG_T_1",
    "ICF_SCS_FG_T_1",
    "ICF_TMS_NOX_A",
    "ACC_SNR_AI_1A",
    "ACT_STATUS",
]


def get_env(name: str, default: Optional[str] = None) -> str:
    v = os.environ.get(name)
    return v if v is not None else ("" if default is None else str(default))


def select_run_id(repo: ModelRepository) -> str:
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
    return repo.get_latest_urea_model_run_id(
        experiment_name=experiment, filter_string=filter_string
    )


def download_model(
    repo: ModelRepository, run_id: str, model_name: str = "urea_gp_model"
) -> Path:
    dst = Path("/tmp/mlflow_models") / f"{run_id}_{model_name}"
    path = repo.download_model_from_run_id(
        run_id=run_id, model_name=model_name, target_path=dst
    )
    print(f"[INFO] 모델 다운로드 경로: {path}")
    print("[INFO] 포함 파일 목록:")
    for p in Path(path).rglob("*"):
        print(" -", p)
    return path


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
    # 요구사항: 최근 20초 조회 (초당 1포인트 가정 → 20개)
    window = get_env("INFLUX_WINDOW", "20s")
    limit = int(get_env("INFLUX_LIMIT", "200"))

    client = InfluxDBClient(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        timeout=30,
    )

    query = f'\nSELECT * FROM "{measurement}" WHERE time >= now() - {window} ORDER BY time DESC LIMIT {limit}\n'
    print("[INFO] Influx 쿼리:", query)
    result = client.query(query)
    points = list(result.get_points()) if result else []
    print(f"[INFO] 조회 포인트 수: {len(points)}")

    if not points:
        raise RuntimeError("최근 구간 데이터가 없습니다. 시간창/측정값을 조정하세요.")

    df = pd.DataFrame(points)
    print("[INFO] 원본 데이터프레임:", df.shape)
    print(df.head(3))
    return df


def aggregate_last_20s_to_5s(df: pd.DataFrame) -> pd.DataFrame:
    """최근 20초 데이터를 5초 윈도우로 요약하여 4행 반환.

    - 센서 컬럼: 5초 평균
    - *_status 컬럼: 각 윈도우의 마지막 값
    - _time_gateway: 각 윈도우의 경계 시각(오른쪽 라벨)
    """
    if "time" not in df.columns:
        raise KeyError("Influx 응답에 'time' 컬럼이 없습니다.")

    # 시간 처리 및 정렬 (오름차순 → 그룹핑 안정화)
    ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.copy()
    df["_ts"] = ts
    df = df.dropna(subset=["_ts"]).sort_values("_ts")
    df = df.set_index("_ts")

    # 필요한 8개 컬럼만 추출(없으면 에러)
    needed = [c for c in REQUIRED_COLUMNS if c != "_time_gateway"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"필요 컬럼 누락: {missing}")
    sub = df[needed].copy()

    # 센서/상태 컬럼 구분
    status_cols = [c for c in sub.columns if c.endswith("_status")]
    sensor_cols = [c for c in sub.columns if c not in status_cols]

    # 그룹핑: 5초 그룹, 라벨은 오른쪽 경계
    # 각 그룹에 대해 센서는 평균, 상태는 마지막 값
    df_mean = sub[sensor_cols].resample("5S", label="right", closed="right").mean()

    # 평균값(연속형) 컬럼들에 대해 NaN 윈도우 ffill 처리 및 로그
    for col in df_mean.columns:
        pre_nan_mask = df_mean[col].isna()
        pre_nan_count = int(pre_nan_mask.sum())
        if pre_nan_count > 0:
            df_mean[col] = df_mean[col].ffill()
            post_nan_mask = df_mean[col].isna()
            post_nan_count = int(post_nan_mask.sum())
            filled_count = pre_nan_count - post_nan_count
            print(
                f"[INFO] {col} 5초 평균 NaN 윈도우: {pre_nan_count} → ffill 후 {post_nan_count} (보간된 윈도우: {filled_count})"
            )
            if filled_count > 0:
                filled_times = df_mean.index[pre_nan_mask & ~post_nan_mask].tolist()
                sample = filled_times[:5]
                if len(filled_times) > 5:
                    print(f"[INFO] 보간된 윈도우 예시(최대 5개): {sample} ...")
                else:
                    print(f"[INFO] 보간된 윈도우: {sample}")
    df_last = (
        sub[status_cols].resample("5S", label="right", closed="right").last()
        if status_cols
        else pd.DataFrame(index=df_mean.index)
    )
    # 상태값(범주형) 컬럼들에 대해서도 윈도우가 비어 NaN이면 직전 값으로 ffill
    if not df_last.empty:
        for col in df_last.columns:
            pre_nan_mask = df_last[col].isna()
            pre_nan_count = int(pre_nan_mask.sum())
            if pre_nan_count > 0:
                df_last[col] = df_last[col].ffill()
                post_nan_mask = df_last[col].isna()
                post_nan_count = int(post_nan_mask.sum())
                filled_count = pre_nan_count - post_nan_count
                print(
                    f"[INFO] {col} 5초 마지막값 NaN 윈도우: {pre_nan_count} → ffill 후 {post_nan_count} (보간된 윈도우: {filled_count})"
                )
                if filled_count > 0:
                    filled_times = df_last.index[pre_nan_mask & ~post_nan_mask].tolist()
                    sample = filled_times[:5]
                    if len(filled_times) > 5:
                        print(f"[INFO] 보간된 윈도우 예시(최대 5개): {sample} ...")
                    else:
                        print(f"[INFO] 보간된 윈도우: {sample}")

    agg = pd.concat([df_mean, df_last], axis=1)
    agg.index.name = "_time_gateway"
    agg = agg.reset_index()

    # 최신 4개 윈도우만 남김 (DESC → 상위 4 → 시간순으로 재정렬)
    agg = (
        agg.sort_values("_time_gateway", ascending=False)
        .head(4)
        .sort_values("_time_gateway")
    )

    # 로그 출력
    print("[INFO] 5초 윈도우 요약:")
    print(agg.tail(4))

    # 열 순서 정렬: REQUIRED_COLUMNS 순서 유지(존재하는 것만)
    ordered_cols = [c for c in REQUIRED_COLUMNS if c in agg.columns]
    agg = agg[ordered_cols]
    return agg


def main() -> None:
    print("=" * 60)
    print("MLflow 모델 기반 실시간 추론 테스트 시작")
    print("=" * 60)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        print(f"[INFO] MLFLOW_TRACKING_URI: {tracking_uri}")
    else:
        print(
            "[WARN] MLFLOW_TRACKING_URI가 설정되지 않았습니다. mlflow 기본 설정을 사용합니다."
        )

    repo = ModelRepository(tracking_uri=tracking_uri)

    # 1) RUN 선택
    run_id = select_run_id(repo)
    print(f"[INFO] 사용 RUN_ID: {run_id}")

    # 2) 모델 다운로드
    model_root = download_model(repo, run_id=run_id, model_name="urea_gp_model")

    # 3) 모델 파일 선택 및 로드
    model_file = pick_model_file(model_root)
    model = joblib.load(model_file)
    print(f"[INFO] 모델 로드 완료: {model_file}")

    # 4) Influx 최근 데이터 조회
    df = query_recent_influx()

    # 5) 5초 윈도우 요약(최근 20초 → 4행)
    agg = aggregate_last_20s_to_5s(df)
    print("[INFO] 모델 입력용 요약(열 순서 고정):", agg.shape)
    print(agg)

    # 6) 모델 입력행 만들기: [Hz, O2, Temp] = [SNR_PMP_UW_S_1, BR1_EO_O2_A, ICF_CCS_FG_T_1]
    feature_cols = ["SNR_PMP_UW_S_1", "BR1_EO_O2_A", "ICF_CCS_FG_T_1"]
    missing_feat = [c for c in feature_cols if c not in agg.columns]
    if missing_feat:
        raise KeyError(f"모델 입력 피처 누락: {missing_feat}")

    X_all = agg[feature_cols]
    valid_mask = ~X_all.isna().any(axis=1)
    invalid_times = agg.loc[~valid_mask, "_time_gateway"].tolist()
    if invalid_times:
        print(
            f"[WARN] 결측치로 인해 예측에서 제외된 5초 구간: {len(invalid_times)}건 → {invalid_times}"
        )

    X = X_all.loc[valid_mask].to_numpy(dtype=float)
    valid_times = agg.loc[valid_mask, "_time_gateway"].tolist()
    print("[INFO] 예측 입력 배열 형태:", X.shape)
    print(X)

    # 7) 예측: 5초 윈도우 평균 입력만 사용하여 각 시점의 NOx 평균 예측 (결측 구간 제외)
    if len(X) > 0:
        pred = model.predict(X)
        for t, v in zip(valid_times, pred):
            val = v[0] if hasattr(v, "__len__") else v
            print(f"[RESULT] {t} → NOx mean={float(val):.3f}")
    else:
        print("[WARN] 예측 가능한(결측 없는) 5초 구간이 없습니다.")

    # 결측으로 제외된 구간은 NaN으로 표시
    for t in invalid_times:
        print(f"[RESULT] {t} → NOx mean=NaN (insufficient data)")

    print("\n요약")
    print("- RUN_ID:", run_id)
    print("- 모델 경로:", model_file)
    print("- 입력 요약 행 수:", len(agg))


if __name__ == "__main__":
    main()
