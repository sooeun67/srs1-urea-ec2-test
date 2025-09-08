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
    # 디버그: 윈도우 매핑 정보 출력(원시 행 → 각 5초 윈도우 내 행 개수)
    win_counts = (
        pd.Series(1, index=df.index)
        .resample("5s", label="right", closed="right")
        .sum()
        .fillna(0)
        .astype(int)
    )
    if not win_counts.empty:
        # 최근/지정 구간의 윈도우 매핑 상세 로그 (최대 8개 윈도우)
        idx_sample = win_counts.index[-8:]
        idx_utc = idx_sample
        counts_sample = win_counts.loc[idx_sample].tolist()
        mapping_log = list(zip(idx_utc, counts_sample))
        print("[DEBUG] 5초 윈도우별 원시 행 개수(UTC):", mapping_log)
    # 각 그룹에 대해 센서는 평균, 상태는 마지막 값 (보간 전 원본)
    df_mean_raw = sub[sensor_cols].resample("5s", label="right", closed="right").mean()
    df_last_raw = (
        sub[status_cols].resample("5s", label="right", closed="right").last()
        if status_cols
        else pd.DataFrame(index=df_mean_raw.index)
    )

    # 보간 전 요약 출력
    agg_pre = pd.concat([df_mean_raw, df_last_raw], axis=1)
    agg_pre.index.name = "_time_gateway"
    agg_pre = agg_pre.reset_index()
    try:
        agg_pre["_time_gateway"] = pd.to_datetime(
            agg_pre["_time_gateway"], utc=True, errors="coerce"
        ).dt.tz_convert("UTC")
    except Exception:
        pass
    # 가장 이른 4개 윈도우(예: 05,10,15,20)만 유지
    agg_pre = agg_pre.sort_values("_time_gateway").head(4)
    print("[INFO] 5초 윈도우 요약(보간 전):")
    print(agg_pre.tail(4))

    # 이후 처리용 복사본에 보간 수행
    df_mean = df_mean_raw.copy()

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
                sample_utc = sample
                if len(filled_times) > 5:
                    print(f"[INFO] 보간된 윈도우 예시(최대 5개, UTC): {sample_utc} ...")
                else:
                    print(f"[INFO] 보간된 윈도우(UTC): {sample_utc}")
    df_last = (
        df_last_raw.copy()
        if not df_last_raw.empty
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
                    sample_utc = sample
                    if len(filled_times) > 5:
                        print(
                            f"[INFO] 보간된 윈도우 예시(최대 5개, UTC): {sample_utc} ..."
                        )
                    else:
                        print(f"[INFO] 보간된 윈도우(UTC): {sample_utc}")

    agg = pd.concat([df_mean, df_last], axis=1)
    agg.index.name = "_time_gateway"
    agg = agg.reset_index()

    # Ensure gateway time is displayed in UTC
    try:
        agg["_time_gateway"] = pd.to_datetime(
            agg["_time_gateway"], utc=True, errors="coerce"
        ).dt.tz_convert("UTC")
    except Exception:
        pass

    # 최신 4개 윈도우만 남김 (DESC → 상위 4 → 시간순으로 재정렬)
    # 가장 이른 4개 윈도우(예: 05,10,15,20)만 유지
    agg = agg.sort_values("_time_gateway").head(4)

    # 로그 출력 (보간 후)
    print("[INFO] 5초 윈도우 요약(보간 후):")
    print(agg.tail(4))

    # 열 순서 정렬: REQUIRED_COLUMNS 순서 유지(존재하는 것만)
    ordered_cols = [c for c in REQUIRED_COLUMNS if c in agg.columns]
    agg = agg[ordered_cols]
    return agg


def main() -> None:
    print("🚀" + "=" * 58)
    print("🚀 MLflow 모델 기반 실시간 추론 테스트 시작")
    print("🚀" + "=" * 58)

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

    # 2) 모델 다운로드
    model_root = download_model(run_id=run_id, model_name="urea_gp_model")

    # 3) 모델 파일 선택 및 로드
    model_file = pick_model_file(model_root)
    model = joblib.load(model_file)
    print(f"✅ 모델 로드 완료: {model_file}")

    # 4) Influx 최근 데이터 조회
    df = query_recent_influx()

    # 5) 5초 윈도우 요약(최근 20초 → 4행)
    agg = aggregate_last_20s_to_5s(df)
    print("🧾 모델 입력용 요약(열 순서 고정):", agg.shape)
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
    print("🧮 예측 입력 배열 형태:", X.shape)
    print(X)

    # 7) 예측: 5초 윈도우 평균 입력만 사용하여 각 시점의 NOx 평균 예측 (결측 구간 제외)
    if len(X) > 0:
        pred = model.predict(X)
        for t, v in zip(valid_times, pred):
            val = v[0] if hasattr(v, "__len__") else v
            print(f"🎯 {t} → NOx mean={float(val):.3f}")
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
