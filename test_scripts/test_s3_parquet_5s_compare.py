"""
S3 parquet → 5초 윈도우 요약 후 InfluxDB 결과와 비교

사용법(UTC 고정 구간 예):
  export START_TIME="2025-08-27 00:00:01"  # UTC 시작 시각
  python test_scripts/test_s3_parquet_5s_compare.py

환경변수:
  - S3_URI: S3 객체 경로 (기본: 예시 파일)
  - START_TIME (UTC) 또는 START_TIME_KST (KST): 고정 구간 시작
  - INFLUX_HOST/PORT/USERNAME/PASSWORD/DB/MEASUREMENT (Influx 비교용, 선택)
  - INFLUX_WINDOW (기본 20s), INFLUX_LIMIT (기본 200)

출력:
  - S3 원시 개요, 5초 윈도우 요약(보간 전/후)
  - 선택 시 Influx 동일 구간 요약
  - 시간 키(_time_gateway) 기준으로 컬럼별 차이(절대값) 요약
"""

import os
import io
import gzip
from pathlib import Path
from typing import List, Optional

import boto3
import pandas as pd
import numpy as np

try:
    from influxdb import InfluxDBClient
except Exception:  # Influx 미사용 시 통과
    InfluxDBClient = None  # type: ignore


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


def _read_from_s3(uri: str) -> bytes:
    if uri.startswith("s3://"):
        _, rest = uri.split("s3://", 1)
        bucket, key = rest.split("/", 1)
    else:
        # '/bucket/...' 형태도 허용
        rest = uri.lstrip("/")
        bucket, key = rest.split("/", 1)

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    return body


def read_parquet_from_s3(uri: str) -> pd.DataFrame:
    """S3에서 데이터를 읽어 DataFrame으로 반환합니다.

    우선 parquet으로 시도하고, 실패 시 JSON lines → CSV 순으로 자동 판별합니다.
    파일명이 .gz면 gzip 해제 후 처리합니다.
    """
    import pyarrow.parquet as pq

    raw = _read_from_s3(uri)
    data = raw
    if uri.endswith(".gz"):
        try:
            data = gzip.decompress(raw)
        except Exception:
            data = raw

    # 1) Parquet 시도
    try:
        table = pq.read_table(io.BytesIO(data))
        df = table.to_pandas(types_mapper=pd.ArrowDtype)
        print("📥 로드 형식: parquet")
        return df
    except Exception:
        pass

    # 2) JSON Lines 시도
    try:
        text = data.decode("utf-8", errors="replace")
        df_json = pd.read_json(io.StringIO(text), lines=True)
        if isinstance(df_json, pd.DataFrame) and not df_json.empty:
            print("📥 로드 형식: jsonl")
            return df_json
    except Exception:
        pass

    # 3) CSV 시도
    try:
        text = data.decode("utf-8", errors="replace")
        df_csv = pd.read_csv(io.StringIO(text))
        print("📥 로드 형식: csv")
        return df_csv
    except Exception as e:
        print(f"❌ 지원하지 않는 파일 형식 또는 손상된 파일: {e}")
        raise


def detect_time_column(df: pd.DataFrame) -> str:
    candidates = ["time", "_time", "timestamp", "event_time"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"시간 컬럼을 찾을 수 없습니다. 후보: {candidates}")


def aggregate_5s(df: pd.DataFrame) -> pd.DataFrame:
    time_col = detect_time_column(df)
    ts = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.copy()
    df["_ts"] = ts
    df = df.dropna(subset=["_ts"]).sort_values("_ts").set_index("_ts")

    needed = [c for c in REQUIRED_COLUMNS if c != "_time_gateway"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"필요 컬럼 누락: {missing}")
    sub = df[needed].copy()

    status_cols = [c for c in sub.columns if c.endswith("_status")]
    sensor_cols = [c for c in sub.columns if c not in status_cols]

    # 보간 전 요약
    mean_raw = sub[sensor_cols].resample("5s", label="right", closed="right").mean()
    last_raw = (
        sub[status_cols].resample("5s", label="right", closed="right").last()
        if status_cols
        else pd.DataFrame(index=mean_raw.index)
    )
    agg_pre = pd.concat([mean_raw, last_raw], axis=1)
    agg_pre.index.name = "_time_gateway"
    agg_pre = agg_pre.reset_index()

    print("🧾 S3 5초 윈도우 요약(보간 전, UTC):")
    print(
        agg_pre.sort_values("_time_gateway").head(4)[
            [c for c in REQUIRED_COLUMNS if c in agg_pre.columns]
        ]
    )

    # ffill
    mean = mean_raw.copy()
    for col in mean.columns:
        if mean[col].isna().any():
            pre = int(mean[col].isna().sum())
            mean[col] = mean[col].ffill()
            post = int(mean[col].isna().sum())
            if pre:
                print(f"🛠️ {col} 평균 NaN: {pre} → {post} (ffill: {pre - post})")
    last = last_raw.copy()
    if not last.empty:
        for col in last.columns:
            if last[col].isna().any():
                pre = int(last[col].isna().sum())
                last[col] = last[col].ffill()
                post = int(last[col].isna().sum())
                if pre:
                    print(f"🛠️ {col} 마지막값 NaN: {pre} → {post} (ffill: {pre - post})")

    agg = pd.concat([mean, last], axis=1)
    agg.index.name = "_time_gateway"
    agg = agg.reset_index()
    # UTC 유지
    agg["_time_gateway"] = pd.to_datetime(
        agg["_time_gateway"], utc=True, errors="coerce"
    )
    agg = agg.sort_values("_time_gateway").head(4)

    print("🧾 S3 5초 윈도우 요약(보간 후, UTC):")
    print(agg[[c for c in REQUIRED_COLUMNS if c in agg.columns]])
    return agg[[c for c in REQUIRED_COLUMNS if c in agg.columns]]


def query_influx_and_aggregate(
    start_utc: pd.Timestamp, secs: int = 20
) -> Optional[pd.DataFrame]:
    if InfluxDBClient is None:
        print("ℹ️ influxdb 패키지가 없어 비교를 생략합니다.")
        return None

    host = get_env("INFLUX_HOST", "10.238.27.132")
    port = int(get_env("INFLUX_PORT", "8086"))
    username = get_env("INFLUX_USERNAME", "read_user")
    password = get_env("INFLUX_PASSWORD", "!Skepinfluxuser25")
    database = get_env("INFLUX_DB", "SRS1")
    measurement = get_env("INFLUX_MEASUREMENT", "SRS1")

    client = InfluxDBClient(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        timeout=30,
    )
    end_utc = start_utc + pd.to_timedelta(max(secs - 1, 0), unit="s")
    q = (
        f'SELECT * FROM "{measurement}" '
        f"WHERE time >= '{start_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}' AND time <= '{end_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}' "
        f"ORDER BY time ASC LIMIT 200"
    )
    print("🔎 Influx 쿼리:", q)
    r = client.query(q)
    pts = list(r.get_points()) if r else []
    print(f"📊 Influx 조회 포인트 수: {len(pts)}")
    if not pts:
        return None
    df = pd.DataFrame(pts)
    return aggregate_5s(df)


def main() -> None:
    print("📦 S3 → 5초 윈도우 요약 및 Influx 비교")
    s3_uri = get_env(
        "S3_URI",
        "/zero4-wte-raw-035989641590-2/plantcode=SRS1/ver=1/field=op_sts/Year=2025/MMDD=0827/zero4-wte-ai-kdf-SRS1-op_sts-1-2025-08-27-00-08-47-1389c518-3e20-4c96-82ee-ef058e3f25a4.gz",
    )
    if not s3_uri.startswith("s3://"):
        s3_uri = f"s3://{s3_uri.lstrip('/')}"
    print("🔗 S3_URI:", s3_uri)

    # 시간 범위: START_TIME(UTC 우선) or START_TIME_KST
    window = get_env("INFLUX_WINDOW", "20s").lower().strip()
    secs = 20
    if window.endswith("s"):
        secs = int(window[:-1] or 0)
    elif window.endswith("m"):
        secs = int(window[:-1] or 0) * 60

    start_time_utc = get_env("START_TIME", "").strip()
    start_time_kst = get_env("START_TIME_KST", "").strip()
    if start_time_utc:
        start_dt_utc = pd.to_datetime(start_time_utc, utc=True, errors="coerce")
        print("⏱️ 기준(UTC):", start_dt_utc)
    elif start_time_kst:
        start_dt_utc = (
            pd.to_datetime(start_time_kst).tz_localize("Asia/Seoul").tz_convert("UTC")
        )
        print("⏱️ 기준(KST→UTC):", start_dt_utc)
    else:
        raise ValueError("START_TIME 또는 START_TIME_KST를 지정하세요.")

    # S3 로드 및 필터
    df_raw = read_parquet_from_s3(s3_uri)
    print("🗂️ S3 로드 DF:", df_raw.shape)
    time_col = detect_time_column(df_raw)
    ts = pd.to_datetime(df_raw[time_col], utc=True, errors="coerce")
    df_raw = df_raw.loc[
        ts.between(
            start_dt_utc, start_dt_utc + pd.to_timedelta(max(secs - 1, 0), unit="s")
        )
    ].copy()
    print("📊 S3 필터 후 포인트 수:", len(df_raw))

    # 5초 요약 (S3)
    s3_agg = aggregate_5s(df_raw)

    # Influx 비교(선택)
    infl_agg = query_influx_and_aggregate(start_dt_utc, secs)
    if infl_agg is None:
        return

    # 비교: 시간 키 기준 내부 조인
    join_key = "_time_gateway"
    common_cols = [
        c
        for c in REQUIRED_COLUMNS
        if c != join_key and c in s3_agg.columns and c in infl_agg.columns
    ]
    merged = s3_agg.merge(
        infl_agg, on=join_key, how="inner", suffixes=("_s3", "_influx")
    )
    print("\n📎 비교 결과(UTC, 공통 시간 교집합):", merged.shape)
    if not merged.empty:
        diffs = {}
        for c in common_cols:
            diffs[c] = (merged[f"{c}_s3"] - merged[f"{c}_influx"]).abs()
        diff_df = pd.DataFrame(diffs)
        diff_df.insert(0, join_key, merged[join_key])
        print("\n📐 절대 차이(컬럼별):")
        print(diff_df)
        print("\n🧮 컬럼별 평균 절대오차:")
        print(diff_df[common_cols].mean(numeric_only=True))


if __name__ == "__main__":
    main()
