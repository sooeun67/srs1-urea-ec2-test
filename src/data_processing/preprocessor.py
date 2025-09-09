# ======================
# preprocessor.py
# ======================
"""
Preprocessing utilities for SKEP Urea Control System

본 모듈은 config 기반 전처리 파이프라인을 제공합니다.
- [TEST 전용] 추천값 계산을 위한 제한 ffill (필요 시 full index 생성)
- [TRAIN 전용] 모형 학습 데이터 생성:
    1) 비정상 가동 구간 제외 (INCINERATORSTATUS == 1 의 ±window_min 제외)
       - cfg.exclude_status_value, cfg.exclude_window_min
    2) 변수별 임계값(global_threshold) 밖의 값 → 결측치
    3) 시간 제약 선형보간 (앞/뒤 유효 관측치가 limit_sec 이내인 경우에만 보간)
       - cfg.interpolate_limit_sec, cfg.interpolate_method
    4) (GP 전용) AI 운전 필터 → dedup → sample → dropna → min_samples 체크
Config 의존 항목
----------------
- CommonPreprocessingConfig:
    - column_config: ColumnConfig (필수)
    - resample_sec: int (full index 간격/ffill 계산에 사용)
    - global_threshold: Dict[str, Tuple[float, float]] (__post_init__에서 실컬럼명 키로 완성)
- TrainPreprocessingConfig / GPTrainPreprocessingConfig:
    - start_time / end_time (선택)
    - interpolate_limit_sec, interpolate_method
    - dedup, sample_size, random_state, dropna_required, min_samples
    - logger_cfg (LoggerConfig)
    - exclude_status_value: int  (기본 1)
    - exclude_window_min: int   (분 단위, 기본 40)
- InferPreprocessingConfig:
    - ffill_limit_sec
    - resample_sec

ColumnConfig 의존 컬럼 예시
--------------------------
- col_datetime (필수)
- col_o2, col_hz, col_inner_temp, col_outer_temp, col_nox (임계값 기준)
- col_incinerator_status

사용 예
-------
from config.preprocessing_config import GPTrainPreprocessingConfig, InferPreprocessingConfig
from config.column_config import ColumnConfig
from preprocess import Preprocessor

cc = ColumnConfig(...)
train_cfg = GPTrainPreprocessingConfig(column_config=cc, plant_code="SRS1")
infer_cfg = InferPreprocessingConfig(column_config=cc, plant_code="SRS1")

pp = Preprocessor()

df_train = pp.make_train_dataset(df_raw, train_cfg)
df_test  = pp.make_test_ffill(df_latest, infer_cfg, require_full_index=True)  # True 또는 False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple, List, Dict

import warnings
import numpy as np
import pandas as pd

from utils.logger import get_logger, LoggerConfig
from config.preprocessing_config import (
    CommonPreprocessingConfig,
    TrainPreprocessingConfig,
    GPTrainPreprocessingConfig,
    InferPreprocessingConfig,
)
from config.column_config import ColumnConfig
from config.model_config import GPModelConfig


__all__ = [
    "Preprocessor",
    "GPTrainPreprocessor",
    "LGBMFeaturePreprocessor",
    "LGBMTrainPreprocessor",
]


# ============================================================
# 내부 유틸
# ============================================================
# 초 단위를 pandas 빈도 문자열로 변환 (예: 5 → '5s')
def _freq_from_seconds(sec: int) -> str:
    """
    정수(초) → pandas 빈도 문자열로 변환.

    Parameters
    ----------
    sec : int
        샘플링/리샘플링 간격(초). 5 → '5s'

    Returns
    -------
    str
        pandas 빈도 문자열 (예: '5s')

    Notes
    -----
    - 0 또는 음수 입력은 에러를 발생시킵니다.
    """
    if sec <= 0:
        raise ValueError(f"resample_sec must be > 0, got {sec}")
    return f"{int(sec)}s"


# datetime 컬럼을 pandas datetime 시리즈로 강제 변환
def _datetime_series(df: pd.DataFrame, col_datetime: str) -> pd.Series:
    """
    지정된 datetime 컬럼을 pandas datetime형으로 강제 변환해 반환.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임
    col_datetime : str
        datetime 컬럼명

    Returns
    -------
    pd.Series
        변환된 datetime 시리즈 (타입 보장)

    Notes
    -----
    - 변환 실패 시 pandas가 가능한 범위에서 파싱합니다.
    - 정렬은 수행하지 않으며, 호출 측에서 정렬하는 것을 권장합니다.
    """
    s = pd.to_datetime(df[col_datetime])
    return s


# full index(균일 시간 인덱스) 생성하여 결측 시점 포함 재인덱싱
def _ensure_full_index(
    df: pd.DataFrame,
    col_datetime: str,
    freq: str,
    col_real_row: str,
) -> pd.DataFrame:
    """
    균일 시간 인덱스(full index)를 생성하여 결측 시점 포함 재인덱싱.

    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임(미리 col_datetime 기준 정렬 권장)
    col_datetime : str
        datetime 컬럼명
    freq : str
        타임스텝 간격(ex. '5s', '1min')

    Returns
    -------
    pd.DataFrame
        full index로 재인덱싱된 데이터프레임.
        결측 시점은 NaN이 포함되며, 이후 ffill/bfill 등으로 처리 가능합니다.

    Notes
    -----
    - start~end 범위는 입력 데이터의 최소/최대 시각을 사용합니다.
    - 반환 시 datetime 컬럼을 복원합니다.
    """
    if df.empty:
        out = df.copy()
        out[col_real_row] = False
        return out

    src = df.copy()
    src[col_datetime] = pd.to_datetime(src[col_datetime])
    src = src.sort_values(col_datetime, ignore_index=True)

    start = src[col_datetime].min()
    end = src[col_datetime].max()
    full_index = pd.date_range(start=start, end=end, freq=freq)

    re = src.set_index(col_datetime).reindex(full_index)
    re = re.reset_index().rename(columns={"index": col_datetime})

    orig_ts = pd.Index(src[col_datetime].unique())
    re[col_real_row] = re[col_datetime].isin(orig_ts)

    return re


# 지정 시간 한도 이내에서만 ffill 적용
def _limited_ffill(
    df: pd.DataFrame,
    col_datetime: str,
    freq: str,
    limit_seconds: int,
) -> pd.DataFrame:
    """
    '시간 기준' 제한 ffill 수행.

    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임
    col_datetime : str
        datetime 컬럼명
    freq : str
        데이터의 균일 간격(예: '5s'). limit_seconds를 '행 수'로 변환에 사용
    limit_seconds : int
        허용할 최대 ffill 시간(초)

    Returns
    -------
    pd.DataFrame
        제한 ffill 적용 결과(행 기준 limit=⌊limit_seconds/freq⌋)

    Notes
    -----
    - pandas ffill(limit=N)는 '행 개수' 단위이므로, 초→행 개수로 변환하여 적용합니다.
    - limit_seconds가 0 이하이면 ffill을 적용하지 않습니다.
    """
    out = df.set_index(col_datetime)
    max_rows = int(pd.Timedelta(seconds=int(limit_seconds)) / pd.Timedelta(freq))
    max_rows = max(0, max_rows)
    if max_rows == 0:
        # 제한 0이면 ffill 미적용
        out2 = out.copy()
    else:
        out2 = out.ffill(limit=max_rows)
    return out2.reset_index()


# 앞뒤 유효 관측 시각 차가 limit_sec 이내인 결측점만 선형 보간
def _time_limited_interpolate(
    df: pd.DataFrame,
    col_datetime: str,
    target_cols: Sequence[str],
    limit_sec: int,
    method: str = "time",
) -> pd.DataFrame:
    """
    '앞/뒤 유효 관측 시각 차'가 limit_sec 이내인 결측점만 선형보간으로 채움.

    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임(정렬되지 않아도 내부에서 정렬)
    col_datetime : str
        datetime 컬럼명
    target_cols : Sequence[str]
        보간 대상 컬럼 목록(수치형)
    limit_sec : int
        보간 허용 시간 한도(초): 이전/다음 유효 시각과의 차가 모두 이내여야 함
    method : {'time', 'linear'}, default 'time'
        pandas.Series.interpolate의 방법

    Returns
    -------
    pd.DataFrame
        시간 제약 조건을 만족하는 결측값만 보간된 데이터프레임

    Notes
    -----
    - 'time' 보간이 불가한 경우(예: 동일 타임스탬프 문제) 자동으로 'linear'로 대체합니다.# (수정) ->  동일 timestamp 문제라는 에러로그 띄우고 drop_duplicates 해서 첫행만 남기는걸로
    - 보간은 limit_area='inside'로 양끝 extrapolation을 방지합니다.
    """
    out = df.copy()
    ts = _datetime_series(out, col_datetime)
    out[col_datetime] = ts
    out = out.sort_values(col_datetime).reset_index(drop=True)

    # 동일 timestamp 처리
    if out[col_datetime].duplicated().any():
        warnings.warn(
            "[TRAIN-COMMON] _time_limited_interpolate: 동일 timestamp 감지 → "
            "drop_duplicates(keep='first') 수행",
            RuntimeWarning,
        )
        out = out.drop_duplicates(subset=[col_datetime], keep="first").reset_index(
            drop=True
        )
        ts = out[col_datetime]  # 갱신

    for col in target_cols:
        if col not in out.columns:
            continue
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue

        # 시간축 보간
        s_time_indexed = pd.Series(s.values, index=ts)
        try:
            s_interp = s_time_indexed.interpolate(method=method, limit_area="inside")
        except (
            Exception
        ):  # (수정) ->  동일 timestamp 문제라는 에러로그 띄우고 drop_duplicates 해서 첫행만 남기는걸로
            # (보강) method='time' 실패 시 linear 재시도 (중복 제거 이후에도 예외 시)
            warnings.warn(
                f"[TRAIN-COMMON] interpolate(method='{method}') 실패 → 'linear'로 재시도: col={col}",
                RuntimeWarning,
            )
            # time 보간이 불가한 경우(예: 동일 timestamp 문제) linear로 fallback
            s_interp = s_time_indexed.interpolate(method="linear", limit_area="inside")

        # 인덱스 재정렬
        s_interp_aligned = pd.Series(s_interp.values, index=s.index)

        # 앞/뒤 유효 시각
        last_valid_time = pd.Series(ts.where(s.notna()), index=s.index).ffill()
        next_valid_time = pd.Series(ts.where(s.notna()), index=s.index).bfill()

        # 시간차(sec)
        dt_prev = (ts - last_valid_time).dt.total_seconds()
        dt_next = (next_valid_time - ts).dt.total_seconds()

        # 보간 허용 마스크
        can_interp = ((dt_prev <= limit_sec) & (dt_next <= limit_sec)).fillna(False)

        # 결측 & 허용 → 보간 반영
        out_col = s.copy()
        idx = s.isna() & can_interp
        out_col[idx] = s_interp_aligned[idx]
        out[col] = out_col

    return out


# ============================================================
# EQ_Status 유틸
# ============================================================
def _estimate_step_seconds(ts: pd.Series) -> float:
    """시간 간격(초) 추정: median diff 사용, 실패 시 1.0초."""
    try:
        diffs = ts.sort_values().diff().dropna()
        if diffs.empty:
            return 1.0
        step = diffs.median().total_seconds()
        return float(step) if step and step > 0 else 1.0
    except Exception:
        return 1.0


# ============================================================
# EQ_Status: 드롭 없이 NaN 마스킹 + 플래그 세팅
# ============================================================
def _apply_eq_status_mask(
    df: pd.DataFrame,
    col_datetime: str,
    sensor_cols,
    resample_sec: int,
    shift_sec: int,
    min_nan_block_sec: int,
    logger: logging.Logger,
    cc: ColumnConfig,
    col_flag: str,  # ColumnConfig.col_eq_status_filtered
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    EQ_Status 기반 처리

    단계
    ----
    (A) 조건 1~3 + 시프트: 대상 센서 타깃 컬럼을 NaN 마스킹하고, 새로 NaN이 된 행은 플래그 True
    (B) 연속 NaN ≥ 임계길이: NaN/드롭 변경 없이 플래그만 True
    (C) 모든 EQ==0 강제: NaN/드롭 변경 없이 플래그만 True

    반환
    ----
    df_out : pd.DataFrame
        - 값이 변경된 DataFrame (단, (B)(C)는 값 변경 없음)
        - 플래그 컬럼(cc.col_eq_status_filtered) 포함
    report : Dict[str, Any]
        - 인덱스 리포트: 'idx_eq_status_filtered_nan', 'idx_eq_status_filtered_not_0', 'idx_eq_status_filtered_total'
        - 카운트 등 부가 정보 로그에 남김

    예외
    ----
    - 필수 컬럼 없으면 KeyError 즉시 발생
    """
    # -----------------------------------------
    # 로거 준비 (예: 상위 코드에서 get_logger로 생성한 logger 전달)
    # -----------------------------------------
    # logger = get_logger(LoggerConfig(name="Preprocess", level=10))  # 상위에서 생성 가정

    # ---------------------------------
    # 사전 준비: 플래그 컬럼 보장
    # ---------------------------------
    out = df.copy()

    col_flag = cc.col_eq_status_filtered
    if col_flag not in out.columns:
        out[col_flag] = False

    n = len(out)

    logger.info("┌─[EQ] 1. EQ_Status 기반 row filtering 시작")

    # ---------------------------------
    # 0) 필수 컬럼 유효성 체크
    # ---------------------------------
    cols_tms_eq_status = list(cc.cols_tms_eq_status)
    cols_tms_value = list(cc.cols_tms_value)
    cols_icf_tms = list(cc.cols_icf_tms)

    missing = [
        c
        for c in (cols_tms_eq_status + cols_tms_value + cols_icf_tms)
        if c not in out.columns
    ]
    if missing:
        raise KeyError(f"[EQ] 필수 컬럼 누락: {missing}")

    # 센서 → 타깃 컬럼 매핑 (각 EQ 상태 컬럼마다 값/ICF 컬럼)
    sensor_cols: Dict[str, List[str]] = {
        eq_col: [val_col, icf_col]
        for eq_col, val_col, icf_col in zip(
            cols_tms_eq_status, cols_tms_value, cols_icf_tms
        )
    }
    window_rows_eq_status_shift = int(shift_sec / resample_sec)
    window_rows_eq_status_min_nan_len = int(min_nan_block_sec / resample_sec)

    # ---------------------------------
    # A) 조건 1~3 + 시프트: NaN 마스킹 + 플래그
    # ---------------------------------
    logger.info("│  ├─[A] EQ_Status 조건(결측/!=0/==1 이후 shift) NaN 마스킹 시작")
    for col in cols_tms_eq_status:
        # 상태/타깃 검증
        if col not in out.columns:
            raise KeyError(f"[EQ] EQ 상태 컬럼 누락: {col}")
        targets = sensor_cols.get(col, [])
        if not targets:
            raise KeyError(f"[EQ] 타깃 컬럼 매핑 없음: {col}")
        missing_targets = [t for t in targets if t not in out.columns]
        if missing_targets:
            raise KeyError(f"[EQ] 타깃 컬럼 누락: {missing_targets} (for {col})")

        status = out[col]
        cond_nan = status.isna()
        cond_non_zero = status.notna() & (status != 0)
        cond_1 = status == 1

        # 보고용 로그
        logger.info(
            "│  │  [%s] NaN:%d / !=0:%d / ==1:%d",
            col,
            int(cond_nan.sum()),
            int(cond_non_zero.sum()),
            int(cond_1.sum()),
        )

        # 위치 기반 시프트(안전)
        pos_1 = np.flatnonzero(cond_1.to_numpy())
        post_mask = np.zeros(n, dtype=bool)
        if window_rows_eq_status_shift > 0 and pos_1.size > 0:
            ends = np.minimum(pos_1 + window_rows_eq_status_shift, n - 1)
            for s, e in zip(pos_1, ends):
                post_mask[s : e + 1] = True

        full_mask = cond_nan.to_numpy() | cond_non_zero.to_numpy() | post_mask
        logger.info("│  │   이후 shift 포함 총 마스킹 수: %d", int(full_mask.sum()))

        # 마스킹 전/후 비교로 "새로 NaN된 행" → 플래그 True
        before_isna = out[targets].isna()
        out.loc[full_mask, targets] = np.nan
        after_isna = out[targets].isna()
        newly_masked_rows = (after_isna & ~before_isna).any(axis=1)

        # 플래그 반영
        out.loc[newly_masked_rows.index[newly_masked_rows], col_flag] = True

    # ---------------------------------
    # B) 연속 NaN ≥ 임계길이: 플래그만 True (값 변경 없음)
    # ---------------------------------
    logger.info(
        "│  ├─[B] 연속 NaN ≥ %d행 플래그 처리 시작",
        int(window_rows_eq_status_min_nan_len),
    )
    all_sensor_cols = sum(sensor_cols.values(), [])
    missing_targets_all = [t for t in all_sensor_cols if t not in out.columns]
    if missing_targets_all:
        raise KeyError(f"[EQ] 타깃 컬럼 누락: {missing_targets_all} (연속 NaN 판정용)")

    nan_all_mask = out[all_sensor_cols].isna().all(axis=1)
    group = (nan_all_mask != nan_all_mask.shift()).cumsum()
    run_sizes = nan_all_mask.groupby(group).transform("size")
    long_nan = nan_all_mask & (run_sizes >= int(window_rows_eq_status_min_nan_len))

    # 값 변경 없이 플래그만
    out.loc[long_nan, col_flag] = True

    idx_eq_status_filtered_nan = out.index[long_nan]
    logger.info("│  │  연속 NaN(임계길이↑) 수: %d", int(long_nan.sum()))

    # ---------------------------------
    # C) 모든 EQ==0만 허용: 플래그만 True (값 변경 없음)
    # ---------------------------------
    logger.info("│  └─[C] 모든 EQ_Status == 0 강제 플래그 처리 시작")
    missing_eq_cols = [c for c in cols_tms_eq_status if c not in out.columns]
    if missing_eq_cols:
        raise KeyError(f"[EQ] EQ 상태 컬럼 누락: {missing_eq_cols}")

    eq_mask = (out[cols_tms_eq_status] == 0).all(axis=1)
    not_all_zero = ~eq_mask
    out.loc[not_all_zero, col_flag] = True

    idx_eq_status_filtered_not_0 = out.index[not_all_zero]
    logger.info("│     EQ!=0 포함 수: %d", int(not_all_zero.sum()))

    # ---------------------------------
    # 최종 요약
    # ---------------------------------
    idx_eq_status_filtered_total = idx_eq_status_filtered_nan.union(
        idx_eq_status_filtered_not_0
    )

    logger.info("├─[EQ] 요약")
    logger.info("│  (참고) shift 행수: %d", int(window_rows_eq_status_shift))
    logger.info(
        "│  (참고) 연속 NaN 임계 행수: %d", int(window_rows_eq_status_min_nan_len)
    )
    logger.info("│  (1) 연속 NaN: %d", len(idx_eq_status_filtered_nan))
    logger.info("│  (2) EQ!=0 : %d", len(idx_eq_status_filtered_not_0))
    logger.info("│  (1)+(2)  : %d", len(idx_eq_status_filtered_total))
    logger.info("└─[EQ] 완료")
    return out


# ============================================================
# INCINERATOR_STATUS: ±window 플래그만 세팅 (nan으로 값 변경 없음)
# ============================================================
def _apply_inc_status_mask(
    df: pd.DataFrame,
    *,
    col_datetime: str,
    col_inc_status: str,
    abnormal_value: int,
    resample_sec: int,
    window_min: int,
    logger: logging.Logger,
    col_flag: str,  # ColumnConfig.col_inc_status_filtered
) -> pd.DataFrame:
    """
    INCINERATOR_STATUS가 abnormal_value인 지점들을 중심으로
    ±(window_min 분) ≒ ±window_rows 행 범위를 '행 기준'으로 계산.
    - 값은 변경하지 않음
    - 해당 구간(팽창 구간)에 대해 col_flag=True 세팅만 수행

    Notes
    -----
    - 시간 기반 병합(타임스탬프 비교) 사용하지 않음.
    - 컬럼 누락 시 KeyError.
    """
    out = df.copy()
    out[col_datetime] = pd.to_datetime(out[col_datetime])
    out = out.sort_values(col_datetime).reset_index(drop=True)
    n = len(out)

    # 플래그 컬럼 보장
    if col_flag not in out.columns:
        out[col_flag] = False

    # 필수 컬럼 검증
    if n == 0:
        logger.info("[INC] 빈 DataFrame → 스킵")
        return out
    if col_inc_status not in out.columns:
        raise KeyError(f"[INC] 상태 컬럼 누락: {col_inc_status}")
    if col_datetime not in out.columns:
        raise KeyError(f"[INC] 시간 컬럼 누락: {col_datetime}")

    # 상태 마스크
    mask_bad = out[col_inc_status] == abnormal_value
    if not mask_bad.any():
        logger.info("[INC] abnormal=%s 행 없음 → 스킵", abnormal_value)
        return out

    # ----- 행 기준 윈도우 크기 계산 -----
    if resample_sec is None or resample_sec <= 0:
        raise ValueError(f"[INC] resample_sec가 유효하지 않습니다: {resample_sec}")
    window_rows = int(round((window_min * 60) / resample_sec))
    window_rows = max(0, window_rows)

    logger.info(
        "[INC] abnormal=%s, window_min=%d, resample_sec=%d → ±%d행(총 %d행 커널)",
        abnormal_value,
        int(window_min),
        int(resample_sec),
        window_rows,
        2 * window_rows + 1,
    )

    # ----- 행 기준 팽창 마스크 (convolution) -----
    v = mask_bad.to_numpy(dtype=int)
    if window_rows == 0:
        window_mask = v.astype(bool)
    else:
        kernel = np.ones(2 * window_rows + 1, dtype=int)
        window_mask = np.convolve(v, kernel, mode="same") > 0

    # ----- 값 변경 없이 플래그만 세팅 -----
    prev_true = out[col_flag].sum()
    out.loc[window_mask, col_flag] = True
    flagged = int(out[col_flag].sum() - prev_true)

    logger.info(
        "[INC] 플래그 세팅 행수=%d (팽창 마스크 총합=%d)",
        flagged,
        int(window_mask.sum()),
    )
    return out


# ============================================================
# 글로벌 임계값: 플래그만 세팅 (nan으로 값 변경 없음)
# ============================================================
def _apply_global_threshold_mask(
    df: pd.DataFrame,
    *,
    thresholds: Dict[str, Tuple[float, float]],
    numeric_cols: Sequence[str],
    col_flag: str,  # ColumnConfig.col_glob_threshold_filtered
) -> pd.DataFrame:
    """
    thresholds에 따라 numeric_cols에서 임계범위 밖의 값이 있는 행을 탐지.
    - 값은 변경하지 않음
    - 해당 행에 대해 col_flag=True 세팅만 수행
    """
    out = df.copy()
    if len(out) == 0:
        if col_flag not in out.columns:
            out[col_flag] = False
        return out

    if col_flag not in out.columns:
        out[col_flag] = False

    # 유효 numeric 컬럼만 사용
    valid_cols = [
        c
        for c in numeric_cols
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c])
    ]
    if not valid_cols:
        return out

    # 컬럼별 임계 밖 여부 계산 → 행 단위 합집합
    flag_mask = pd.Series(False, index=out.index)
    for col in valid_cols:
        lo, hi = thresholds.get(col, (0.0, np.inf))
        s = out[col]
        # "마스킹 대상"은 정상범위 밖이면서 값이 존재하는 경우로 정의
        out_of_range = s.notna() & ~s.between(lo, hi)
        flag_mask |= out_of_range

    out.loc[flag_mask, col_flag] = True
    return out


# ============================================================
# Public API
# ============================================================
@dataclass
class Preprocessor:
    """
    Config-driven 전처리기.

    Methods
    -------
    make_test_ffill(df, cfg, require_full_index=False, freq=None)
        - 제한 ffill 데이터셋 생성 (추천값 계산용)
    make_train_dataset(df, cfg)
        - 학습용 데이터셋 생성:
          시간필터 → 비정상제외 → 이상치→NaN → 시간제약보간
    """

    # 공통 스키마/룰
    column_config: ColumnConfig = field(default_factory=ColumnConfig)

    # 전처리 설정(학습/추론)
    model_config: GPModelConfig = field(default_factory=GPModelConfig)
    prep_cm_train_cfg: TrainPreprocessingConfig = field(
        default_factory=TrainPreprocessingConfig
    )
    prep_gp_train_cfg: GPTrainPreprocessingConfig = field(
        default_factory=GPTrainPreprocessingConfig
    )
    prep_infer_cfg: InferPreprocessingConfig = field(
        default_factory=InferPreprocessingConfig
    )

    # -----------------------------
    # [TRAIN] 학습 데이터 생성 (공통 전처리)
    # -----------------------------
    def make_train_dataset(
        self,
        df: pd.DataFrame,
        *,
        logger_cfg: Optional[LoggerConfig] = None,
    ) -> pd.DataFrame:
        """
        학습용 데이터셋 생성 파이프라인:
        - config 기반으로 이상치 제거, 결측 보간, 비정상 상태 구간 제외 등을 수행하여
          모델 학습에 적합한 데이터셋을 반환합니다.

        처리 순서
        ----------
        (선행) 시간 필터: start_time ~ end_time 범위로 데이터 제한 -> 현재 안 함
        # 추후 S3 에서 가져올 때부터 시간으로 필터링 해서 가져오는 걸 가정하고 작성 ============
        st = getattr(cfg, "start_time", None)
        et = getattr(cfg, "end_time", None)
        out[col_datetime] = _datetime_series(out, col_datetime)
        out = out.sort_values(col_datetime).reset_index(drop=True)
        if st or et:
            lg.info(f"[TRAIN-COMMON] 시간 필터 적용: start={st}, end={et}")
            if st: out = out.loc[out[col_datetime] >= pd.to_datetime(st)]
            if et: out = out.loc[out[col_datetime] <= pd.to_datetime(et)]
            out = out.reset_index(drop=True)
            lg.debug(f"[TRAIN-COMMON] 시간 필터 후 행수: {len(out):,}")
        if out.empty:
            lg.error("[TRAIN-COMMON] 시간 필터 결과가 비었습니다.")
            return out
        # 추후 S3 에서 가져올 때부터 시간으로 필터링 해서 가져오는 걸 가정하고 작성 ============
        0) 시간 reindexing (full index) 수행
        1) EQ_Status 기반 row filtering
        2) 소각로 비정상 가동 구간 제외
        3) 임계값 기반 이상치 처리: out-of-range → NaN
        4) 시간 제약 보간: 앞/뒤 유효 관측이 limit_sec 이내인 경우만 보간
        5) 시간 reindexing으로 생긴 rows 제외(col_real_row == True만 유지)

        Parameters
        ----------
        df : pd.DataFrame
            원본 데이터프레임
        logger_cfg : LoggerConfig, optional
            별도의 로깅 설정을 적용할 경우 지정

        Returns
        -------
        pd.DataFrame
            학습에 적합하도록 전처리된 데이터프레임.
            - 시간 필터, 비정상 구간 제외, 임계값 기반 NaN 처리, 시간 제약 선형보간 후 최종 데이터 반환
        """
        tc = self.prep_cm_train_cfg
        cc = self.column_config
        mc = self.model_config
        lg = get_logger(
            logger_cfg
            or getattr(
                tc, "logger_cfg", LoggerConfig(name="Preprocess.TRAIN", level=10)
            )
        )

        col_datetime = cc.col_datetime
        col_real_row = cc.col_real_row
        out = df.copy()
        if out.empty:
            lg.warning("[TRAIN-COMMON] 입력 df가 비어있습니다. 그대로 반환합니다.")
            return out

        # 필요한 경우 시간 필터는 상위 I/O에서 처리
        out[col_datetime] = _datetime_series(out, col_datetime)
        out = out.sort_values(col_datetime).reset_index(drop=True)

        # 0) 시간 reindexing (full index)
        f = _freq_from_seconds(int(tc.resample_sec))
        before_n = len(out)
        out = _ensure_full_index(out, col_datetime, freq=f, col_real_row=col_real_row)
        after_n = len(out)
        added_n = after_n - before_n
        lg.info(
            "[TRAIN-COMMON][FULL-INDEX] 행수: %s → %s (추가 %s)",
            f"{before_n:,}",
            f"{after_n:,}",
            f"{added_n:,}",
        )

        # 1) EQ_Status 기반 row filtering (col_real_row와 무관)
        out = _apply_eq_status_mask(
            df=out,
            col_datetime=col_datetime,
            sensor_cols=cc.eq_map,  # ← eq_map 사용
            resample_sec=int(tc.resample_sec),
            shift_sec=int(tc.eq_shift_sec),  # ← eq_shift_sec 사용
            min_nan_block_sec=int(tc.eq_min_nan_block_sec),  # ← eq_min_nan_block_sec
            logger=lg,
            cc=cc,
            col_flag=cc.col_eq_status_filtered,
        )

        # 2) 소각로 비정상 가동 구간 제외 (col_real_row와 무관)
        col_inc_status = cc.col_inc_status
        before = len(out)
        out = _apply_inc_status_mask(
            df=out,
            col_datetime=col_datetime,
            col_inc_status=col_inc_status,
            abnormal_value=tc.exclude_status_value,
            resample_sec=int(tc.resample_sec),
            window_min=tc.exclude_window_min,
            logger=lg,
            col_flag=cc.col_inc_status_filtered,
        )
        lg.info(
            "[TRAIN-COMMON] 비정상 상태: %s == %s",
            col_inc_status,
            tc.exclude_status_value,
        )
        lg.info(
            "[TRAIN-COMMON] 비정상 범위: 비정상 상태 rows + 앞뒤 %d분",
            tc.exclude_window_min,
        )
        lg.info(
            "[TRAIN-COMMON] 비정상 범위 제외: %s → %s", f"{before:,}", f"{len(out):,}"
        )
        if out.empty:
            lg.error("[TRAIN-COMMON] 비정상 제외 후 데이터가 비었습니다.")
            return out

        # 3) 임계값 밖 NaN
        out = _apply_global_threshold_mask(
            df=out,
            thresholds=tc.global_threshold,
            numeric_cols=tc._numeric_var_keys,
            col_flag=cc.col_glob_threshold_filtered,
        )

        lg.info("[TRAIN-COMMON] 임계값 기반 이상치 마스킹 완료")

        # 4) 시간 제약 보간
        out = _time_limited_interpolate(
            df=out,
            col_datetime=col_datetime,
            target_cols=mc._training_required(),
            limit_sec=int(tc.interpolate_limit_sec),
            method=str(tc.interpolate_method),
        )
        lg.info(
            "[TRAIN-COMMON] 시간 제약 보간 완료: method='%s', limit_sec=%s",
            tc.interpolate_method,
            tc.interpolate_limit_sec,
        )

        # === [FLAGS FILTER] 보간 후: 플래그 3종의 합집합(True가 하나라도 있으면 제외)만들어 필터 ===
        flag_cols = [
            cc.col_eq_status_filtered,
            cc.col_inc_status_filtered,
            cc.col_glob_threshold_filtered,
        ]

        # 필수 플래그 컬럼 존재 검증 (없으면 즉시 에러)
        missing_flags = [c for c in flag_cols if c not in out.columns]
        if missing_flags:
            raise KeyError(f"[TRAIN-COMMON][FLAGS] 플래그 컬럼 누락: {missing_flags}")

        # 안전: bool 변환 및 결측 False 처리
        flags_df = out[flag_cols].astype(bool).fillna(False)

        # 로그를 위해 필터 전 집계
        before_flags = len(out)
        flag_counts = {c: int(flags_df[c].sum()) for c in flag_cols}
        any_flag_true = flags_df.any(axis=1)
        n_any_true = int(any_flag_true.sum())

        # all False (== 어떤 필터에도 걸리지 않음) 행만 유지
        keep_mask = ~any_flag_true
        out = out.loc[keep_mask].copy()
        after_flags = len(out)

        lg.info(
            "[TRAIN-COMMON][FLAGS] 합집합 제외 적용: %s → %s (제거 %s)",
            f"{before_flags:,}",
            f"{after_flags:,}",
            f"{(before_flags - after_flags):,}",
        )
        lg.info(
            "[TRAIN-COMMON][FLAGS] 상세: any(True)=%s, %s=%d, %s=%d, %s=%d",
            f"{n_any_true:,}",
            cc.col_eq_status_filtered,
            flag_counts[cc.col_eq_status_filtered],
            cc.col_inc_status_filtered,
            flag_counts[cc.col_inc_status_filtered],
            cc.col_glob_threshold_filtered,
            flag_counts[cc.col_glob_threshold_filtered],
        )

        # 5) 시간 reindexing으로 생긴 rows 제외 (real row만 유지)
        before_real_filter = len(out)
        if col_real_row not in out.columns:
            raise ValueError(
                f"[TRAIN-COMMON] '{col_real_row}' 컬럼이 없습니다. full index 생성 단계를 확인하세요."
            )
        out = out.loc[out[col_real_row].astype(bool)].copy()
        after_real_filter = len(out)
        removed_fake = before_real_filter - after_real_filter
        lg.info(
            "[TRAIN-COMMON][REAL-ONLY] 삽입행 제거: %s → %s (제거 %s)",
            f"{before_real_filter:,}",
            f"{after_real_filter:,}",
            f"{removed_fake:,}",
        )

        lg.info("[TRAIN-COMMON] 공통 전처리 완료 (최종 행수: %s)", f"{len(out):,}")
        return out

    # -----------------------------
    # [TEST] 추천값 계산용 ffill
    # -----------------------------
    def make_infer_ffill(
        self,
        df: pd.DataFrame,
        *,
        require_full_index: bool = True,
        logger_cfg: Optional[LoggerConfig] = None,
    ) -> pd.DataFrame:
        """
        추천 계산용 테스트 데이터셋 생성:
        - 주어진 raw 데이터프레임에 대해 제한된 ffill을 수행하여 추론 시 입력 데이터 품질을 보장합니다.

        처리 순서
        ----------
        1) datetime 컬럼을 pandas datetime으로 변환 및 정렬
        2) (옵션) full index 생성: 시간축이 균일하지 않은 경우 full index를 생성하여 결측 시점 포함
        3) 제한 ffill 수행: config.ffill_limit_sec 이하의 결측만 forward-fill로 채움

        Parameters
        ----------
        df : pd.DataFrame
            추천 계산용 원본 데이터프레임(최근 구간 데이터)
        require_full_index : bool, default True
            True → full index 생성 후 ffill 수행
            False → 입력 데이터가 이미 균일 간격이라고 가정하고 ffill만 수행
        freq : str, optional
            full index 간격(예: '5s'). 미지정 시 cfg.resample_sec 사용
        logger_cfg : LoggerConfig, optional
            별도의 로깅 설정을 적용할 경우 지정

        Returns
        -------
        pd.DataFrame
            제한 ffill이 적용된 테스트용 데이터프레임
            - col_datetime 기준 정렬 완료
            - (옵션) full index 적용
            - 제한 조건에 따라 NaN 일부 잔존 가능
        """
        ic = self.prep_infer_cfg
        tc = self.prep_infer_cfg
        cc = self.column_config
        # 로거 우선순위: 인자 logger_cfg > cfg.logger_cfg > 기본값
        lg = get_logger(
            logger_cfg
            or getattr(ic, "logger_cfg", LoggerConfig(name="Preprocess.TEST", level=10))
        )

        col_datetime = cc.col_datetime
        col_real_row = cc.col_real_row
        out = df.copy()
        if out.empty:
            lg.warning("[TEST] 입력 df가 비어있습니다. 그대로 반환합니다.")
            return out

        # 1) datetime 보장 및 정렬
        out[col_datetime] = _datetime_series(out, col_datetime)
        out = out.sort_values(col_datetime).reset_index(drop=True)
        lg.debug(f"[TEST] 입력 행수: {len(out):,}")

        # 2) full index (옵션)
        f = _freq_from_seconds(tc.resample_sec)
        if require_full_index:
            lg.info(f"[TEST] full index 생성: freq='{f}'")
            out = _ensure_full_index(
                out, col_datetime, freq=f, col_real_row=col_real_row
            )
            lg.debug(f"[TEST] full index 후 행수: {len(out):,}")

        # 3) 제한 ffill
        lg.info(f"[TEST] 제한 ffill: limit_sec={ic.ffill_limit_sec}, freq='{f}'")
        out = _limited_ffill(out, col_datetime, f, ic.ffill_limit_sec)

        lg.info("[TEST] ffill 완료")
        return out

    # # -----------------------------------------
    # # 학습 전용: 데이터 추가 전처리리
    # # -----------------------------------------
    # @dataclass
    # class GPTrainPreprocessor:
    #     # 공통 스키마/룰
    #     column_config: ColumnConfig = field(default_factory=ColumnConfig)

    #     # 전처리 설정(학습/추론)
    #     model_config: GPModelConfig = field(default_factory=GPModelConfig)
    #     prep_cm_train_cfg: TrainPreprocessingConfig = field(default_factory=TrainPreprocessingConfig)
    #     prep_gp_train_cfg: GPTrainPreprocessingConfig = field(default_factory=GPTrainPreprocessingConfig)
    # -----------------------------
    # [TRAIN] GP 전용 추가 전처리: AI운전여부 필터링 → 중복행 제거 → 샘플링 → 결측치 제거 → 최소 유효샘플 개수 체크
    # -----------------------------
    def make_train_gp(
        self,
        df: pd.DataFrame,
        *,
        logger_cfg: Optional[LoggerConfig] = None,
    ) -> pd.DataFrame:
        """
        GP 모델 학습 전처리(공통 + GP 추가 단계):
        - 공통 전처리(make_train_common) 수행 후,
        - 1) AI 운전 여부 필터링: ColumnConfig.col_ai == 1 인 행만 남김
        - 2) 중복행 제거: dedup (subset=핵심 피처)
        - 3) 샘플링: sample (n=gc.sample_size, rs=gc.random_state)
        - 4) 결측치 제거: dropna (subset=핵심 피처, gc.dropna_required)
        - 5) 최소 유효샘플 개수 체크: min_samples 경고
        """
        gc = self.prep_gp_train_cfg
        tc = self.prep_cm_train_cfg
        cc = self.column_config
        mc = self.model_config
        lg = get_logger(
            logger_cfg
            or getattr(gc, "logger_cfg", LoggerConfig(name="Preprocess.GP", level=10))
        )
        col_datetime = cc.col_datetime

        # ------------------------------------------------------------------
        # 0) (중요) 공통 전처리 제거
        #    입력 df는 이미 공통 전처리(make_train_dataset) 완료본이어야 함
        # ------------------------------------------------------------------
        out = df.copy()
        lg.info("[TRAIN-GP] GP 전용 전처리 시작 (입력=공통 전처리 완료본 가정)")
        lg.debug("[TRAIN-GP] 입력 크기=%s", out.shape)

        # (안전) datetime 정렬 보장
        if col_datetime in out.columns:
            out[col_datetime] = _datetime_series(out, col_datetime)
            out = out.sort_values(col_datetime).reset_index(drop=True)

        # ------------------------------------------------------------------
        # 1) AI 운전 필터 (필수 단계)
        #    - 반드시 ColumnConfig.col_ai 가 정의되어 있어야 하며,
        #      입력 df 에 해당 컬럼이 존재해야 합니다.
        #    - 조건: df[cc.col_ai] == 1 인 행만 유지
        #    - 미존재 시, 명시적 에러로 중단(설정/스키마 보완 유도)
        # ------------------------------------------------------------------
        if not hasattr(cc, "col_ai") or cc.col_ai not in out.columns:
            raise ValueError(
                "[TRAIN-GP] ColumnConfig.col_ai 누락 또는 입력 데이터에 없음."
            )
        before_ai = len(out)
        out = out.loc[out[cc.col_ai] == 1].copy()
        lg.info(
            "[TRAIN-GP] AI 운전 필터 적용: %d → %d (조건: %s == 1)",
            before_ai,
            len(out),
            cc.col_ai,
        )
        if out.empty:
            raise ValueError("[TRAIN-GP] AI 운전(=1) 구간 없음.")

        # ------------------------------------------------------------------
        # 2) dedup / 3) sample / 4) dropna / 5) min_samples
        #    - 핵심 피처(subset) = ColumnConfig 기준 필수 피처 집합(존재 컬럼만)
        # ------------------------------------------------------------------
        req_cols = mc._training_required()
        # # 실제로 존재하는 컬럼만 필터
        # req_cols = [c for c in req_cols if c in out.columns]
        lg.debug("[TRAIN-GP] 필수 컬럼=%s", req_cols)
        missing = [c for c in req_cols if c not in out.columns]
        if missing:
            msg = f"필수 컬럼 누락: {missing} (요구: {req_cols})"
            lg.error("[TRAIN-GP] %s", msg)
            raise ValueError(msg)

        # 2) 중복행 제거: dedup
        if gc.dedup and req_cols:
            before = len(out)
            out = out.drop_duplicates(subset=req_cols, keep="first")
            lg.info(
                f"[TRAIN-GP] 중복 제거(subset={req_cols}): {before:,} → {len(out):,} 행"
            )

        # 3) 샘플링: sample
        if gc.sample_size is not None and len(out) > int(gc.sample_size):
            before = len(out)
            rs = int(getattr(gc, "random_state", 42))
            out = (
                out.sample(n=int(gc.sample_size), random_state=rs)
                .sort_values(col_datetime)
                .reset_index(drop=True)
            )
            lg.info(
                f"[TRAIN-GP] 샘플링: {before:,} → {len(out):,} (n={gc.sample_size}, rs={gc.random_state})"
            )

        # 4) 결측치 제거: dropna
        if gc.dropna_required and req_cols:
            before = len(out)
            out = out.dropna(subset=req_cols)
            lg.info(
                f"[TRAIN-GP] 결측 행 제거(subset={req_cols}): {before:,} → {len(out):,} 행"
            )

        # 5) 최소 샘플 확보 확인: min_samples
        if len(out) < int(gc.min_samples):
            lg.warning(f"[TRAIN-GP] min_samples 미만: {len(out)} < {gc.min_samples}.")
            lg.warning(f"학습에 부적합할 수 있습니다.")
        lg.info("[TRAIN-GP] GP 전용 전처리 완료")
        return out


"""
LightGBM 추가

고민, 2025-09-08
- train 외 infer용 preprocessing 필요할지? (`act_status` 등)
- GP는 class 안에 method로 처리하는데, LGBM은 class 별도로 생성
- logger 추가 필요
    - (As-Is) print
    - (To-Do) logger
"""

# === Local application ===
from config.preprocessing_config import (
    _LGBMWindowsMixin,
    LGBMTrainPreprocessingConfig,
    LGBMInferPreprocessingConfig,
)

# === Helper 함수 ===


def generate_interval_summary_features_time(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    columns: Sequence[str],
    windows_summary_sec: Sequence[int],
    windows_rate_sec: Sequence[int],
    coerce_numeric: bool = True,
    return_df: bool = True,
    rolling_closed: str | None = None,
    rolling_min_periods: int | None = None,
):
    df_work = df.sort_values(datetime_col).copy()
    t = pd.to_datetime(df_work[datetime_col].values)

    # 원본과 동등한 Δt 계산
    dt_sec = pd.Series(t, index=t).diff().dt.total_seconds().replace(0, np.nan)

    new_columns: list[str] = []

    for col in columns:
        s_raw = (
            pd.to_numeric(df_work[col], errors="coerce")
            if coerce_numeric
            else df_work[col]
        )
        s_val = pd.Series(s_raw.to_numpy(), index=t)

        rate_per_sec = s_val.diff() / dt_sec

        out_chunks: list[pd.DataFrame] = []

        # (1) mean/std
        for sec in windows_summary_sec or []:
            win = f"{sec}s"
            s_mean = (
                s_val.rolling(
                    win, closed=rolling_closed, min_periods=rolling_min_periods
                )
                .mean()
                .reindex(t)
            )
            s_std = (
                s_val.rolling(
                    win, closed=rolling_closed, min_periods=rolling_min_periods
                )
                .std()
                .reindex(t)
            )

            feat = pd.DataFrame(
                {
                    f"{col}_mean_{sec}s": s_mean.values,
                    f"{col}_std_{sec}s": s_std.values,
                },
                index=df_work.index,
            )
            out_chunks.append(feat)
            new_columns += list(feat.columns)

        # (2) momentum max/min
        for sec in windows_rate_sec or []:
            win = f"{sec}s"
            mom_up = (
                rate_per_sec.rolling(
                    win, closed=rolling_closed, min_periods=rolling_min_periods
                )
                .max()
                .reindex(t)
            )
            mom_dn = (
                rate_per_sec.rolling(
                    win, closed=rolling_closed, min_periods=rolling_min_periods
                )
                .min()
                .reindex(t)
            )

            feat = pd.DataFrame(
                {
                    f"{col}_momentum_max_up_{sec}s": mom_up.values,
                    f"{col}_momentum_max_down_{sec}s": mom_dn.values,
                },
                index=df_work.index,
            )
            out_chunks.append(feat)
            new_columns += list(feat.columns)

        if out_chunks:
            feat_df = pd.concat(out_chunks, axis=1).replace([np.inf, -np.inf], np.nan)
            df_work.loc[:, feat_df.columns] = feat_df

    return (df_work, new_columns) if return_df else new_columns


def create_spike_weighted_target(
    df: pd.DataFrame,
    *,
    # ---- 입력 컬럼 ----
    time_col: str,
    nox_col: str,
    # ---- 출력 컬럼 ----
    target_col: str = "target",
    weight_col: str = "weights",
    is_spike_col: str = "is_spike",
    flag_interval_hit_col: str = "flag_interval_hit",
    has_target_col: str = "has_target",
    # ---- 파라미터 ----
    delta_sec: int = 150,
    step_sec: int = 5,
    low_thr: float = 20.0,
    high_thr: float = 40.0,
    lookback_sec: int = 150,
    spike_weight: float = 10.0,
    default_weight: float = 1.0,
    return_flags: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    기존 동작 동일하되, 모든 중간/출력 컬럼명을 파라미터로 주입 가능하게 수정.
    반환: (가중치/타깃/플래그가 추가된 df, intervals_df)
    """

    # 기존 보조 열 정리 (존재하면 삭제)
    for col in (
        target_col,
        weight_col,
        is_spike_col,
        flag_interval_hit_col,
        has_target_col,
    ):
        if col in df.columns:
            del df[col]

    # 1) target 생성 (미래값을 현재 시각으로 당겨 붙임)
    target_df = df[[time_col, nox_col]].copy()
    target_df[time_col] = target_df[time_col] - pd.Timedelta(seconds=delta_sec)
    target_df = target_df.rename(columns={nox_col: target_col})
    df = pd.merge(df, target_df, on=time_col, how="left")

    t = pd.to_datetime(df[time_col])
    s_nox = df[nox_col]

    # 2) 연속 > high_thr 블록 찾기
    high_mask = s_nox > high_thr
    high_times = t[high_mask]
    gid = (high_times.diff() > pd.Timedelta(seconds=step_sec)).cumsum()

    lookback = pd.Timedelta(seconds=lookback_sec)

    intervals = []  # 원본 구간(시프트 전)
    low_mask = s_nox < low_thr

    # 각 >high_thr 블록마다 스파이크 가중 구간 생성
    if not high_times.empty:
        grouped = pd.Series(high_times.values, index=high_times.values).groupby(
            gid.values
        )
        for _, grp in grouped:
            t_high_first = pd.to_datetime(grp.iloc[0])  # >high_thr 진입 시각
            win_start = t_high_first - lookback
            win_end = t_high_first

            win_mask = (t >= win_start) & (t < win_end) & low_mask
            if not win_mask.any():
                continue

            t_low_first = t[win_mask].iloc[0]
            low_val = (
                float(s_nox.loc[t == t_low_first].iloc[0])
                if (t == t_low_first).any()
                else np.nan
            )

            intervals.append(
                {
                    "t_low_first": t_low_first,
                    "t_high_first": t_high_first,
                    "low_value_at_first": low_val,
                }
            )

    # 3) 가중치/스파이크 부여 (타겟 정렬에 맞춰 구간 시프트)
    df[weight_col] = default_weight
    if return_flags:
        df[flag_interval_hit_col] = False
        df[has_target_col] = df[target_col].notna()

    is_spike = np.zeros(len(df), dtype=bool)

    # intervals_df 구성(시프트 전/후, 길이, 히트 카운트 포함)
    intervals_df = pd.DataFrame(intervals)
    if not intervals_df.empty:
        intervals_df["shifted_start"] = intervals_df["t_low_first"] - pd.Timedelta(
            seconds=delta_sec
        )
        intervals_df["shifted_end"] = intervals_df["t_high_first"] - pd.Timedelta(
            seconds=delta_sec
        )
        intervals_df["duration_sec"] = (
            intervals_df["t_high_first"] - intervals_df["t_low_first"]
        ).dt.total_seconds()

        # 전체 hit 마스크 계산과 동시에 intervals_df에 n_rows_hit 채우기
        hit_global = np.zeros(len(df), dtype=bool)
        n_rows_hit = []
        for _, row in intervals_df.iterrows():
            lo_s = row["shifted_start"]
            hi_s = row["shifted_end"]
            hit_i = (t >= lo_s) & (t <= hi_s)
            n_rows_hit.append(int(hit_i.sum()))
            hit_global |= hit_i

        intervals_df["n_rows_hit"] = n_rows_hit

        # 유효 타겟이 있는 위치만 최종 반영
        valid = df[target_col].notna()
        final = hit_global & valid
        df.loc[final, weight_col] = spike_weight
        is_spike = final

        if return_flags:
            df.loc[hit_global, flag_interval_hit_col] = True

        intervals_df = intervals_df.reset_index(drop=True)
        intervals_df.insert(0, "interval_id", np.arange(len(intervals_df)))
    else:
        intervals_df = pd.DataFrame(
            columns=[
                "interval_id",
                "t_low_first",
                "t_high_first",
                "shifted_start",
                "shifted_end",
                "duration_sec",
                "n_rows_hit",
                "low_value_at_first",
            ]
        )

    # is_spike 컬럼
    df[is_spike_col] = is_spike

    # 4) 요약 출력(옵션)
    if verbose:
        n_nan = int(df[target_col].isna().sum())
        vc = df[weight_col].value_counts().sort_index()
        print("🎯 타겟 생성 완료")
        print(f"   - 결측 타겟: {n_nan:,}개")
        print("⚖️ 가중치 분포:")
        for w, c in vc.items():
            print(f"     가중치 {w}: {c:,}개 ({c/len(df)*100:.1f}%)")
        print(
            f"🔎 {is_spike_col}=True: {int(df[is_spike_col].sum()):,}개 "
            f"({df[is_spike_col].mean()*100:.1f}%)"
        )
        print(f"📦 intervals_df: {len(intervals_df)}개 구간")

    return df, intervals_df


# === 전처리용 Class ===


# -----------------------------
# 공통: 요약통계/모멘텀 생성
# -----------------------------
@dataclass
class LGBMFeaturePreprocessor:
    # cfg는 _LGBMWindowsMixin + CommonPreprocessingConfig를 포함하는 어떤 설정이든 OK
    # (예: LGBMTrainPreprocessingConfig, LGBMInferPreprocessingConfig)
    cfg: "_LGBMWindowsMixin"

    def make_interval_features(
        self,
        df: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
        return_df: bool = True,
        rolling_closed: str | None = None,
        rolling_min_periods: int | None = None,
        coerce_numeric: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        if not columns:
            columns = (
                self.cfg.column_config.lgbm_feature_columns
            )  # 비면 ColumnConfig에서 ValueError

        df2, new_cols = generate_interval_summary_features_time(
            df,
            datetime_col=self.cfg.column_config.col_datetime,
            columns=list(columns),
            windows_summary_sec=self.cfg.windows_summary_sec,
            windows_rate_sec=self.cfg.windows_rate_sec,
            coerce_numeric=coerce_numeric,
            return_df=return_df,
            rolling_closed=rolling_closed,
            rolling_min_periods=rolling_min_periods,
        )
        return df2, new_cols


# -----------------------------------------
# 학습 전용: 타깃/가중치 + 학습 프레임 정리
# -----------------------------------------
@dataclass
class LGBMTrainPreprocessor:
    cfg: "LGBMTrainPreprocessingConfig"

    def make_spike_weighted_target(
        self,
        df: pd.DataFrame,
        *,
        apply_high_nox_weight: bool = True,
        high_nox_basis: Literal["current", "target"] = "target",
        return_flags: bool = True,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cc = self.cfg.column_config

        out_df, intervals_df = create_spike_weighted_target(
            df=df,  # 직전 단계에서 copy 했다면 여기선 X
            time_col=cc.col_datetime,
            nox_col=cc.col_nox,  # 또는 cc.target_column
            target_col=cc.col_lgbm_tmp_target_shift,
            weight_col=cc.col_lgbm_tmp_weight,
            is_spike_col=cc.col_lgbm_tmp_is_spike,
            flag_interval_hit_col=cc.col_lgbm_tmp_flag_interval_hit,
            has_target_col=cc.col_lgbm_tmp_has_target,
            delta_sec=self.cfg.weight_spike_delta_sec,
            step_sec=self.cfg.weight_spike_step_sec,
            low_thr=self.cfg.weight_spike_thr_low,
            high_thr=self.cfg.weight_spike_thr_high,
            lookback_sec=self.cfg.weight_spike_lookback_sec,
            spike_weight=self.cfg.weight_spike_pos,
            default_weight=self.cfg.weight_spike_neg,
            return_flags=return_flags,
            verbose=verbose,
        )
        return out_df, intervals_df

    def prepare_training_frame(
        self,
        df: pd.DataFrame,
        *,
        stat_feature_cols: Sequence[str],  # 공통 전처리에서 반환된 new_cols
        extra_feature_cols: Sequence[str] = (),  # 필요 시 추가 피처
        drop_missing: bool = True,
        debug_nan: bool = False,
        verbose: bool = True,
        apply_high_nox_weight: bool = True,
        high_nox_basis: Literal["target", "current"] = "target",
    ) -> Tuple[pd.DataFrame, List[str], pd.Series]:
        cc = self.cfg.column_config

        # 1) feature 목록 구성
        base_features: List[str] = list(cc.lgbm_feature_columns)
        feature_cols: List[str] = list(
            dict.fromkeys([*base_features, *stat_feature_cols, *extra_feature_cols])
        )

        # 2) 타깃 NaN 제거
        before_target = df.shape[0]
        df = df.loc[df[cc.col_lgbm_tmp_target_shift].notna(), :].copy()
        after_target = df.shape[0]
        if verbose:
            print(
                f"🎯 타깃 존재 행만 사용: {before_target:,} → {after_target:,} (제거 {before_target - after_target:,})"
            )

        keep_cols = [
            cc.col_datetime,
            cc.col_lgbm_tmp_target_shift,
            cc.col_lgbm_tmp_is_spike,
            cc.col_lgbm_tmp_weight,
        ] + feature_cols

        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise KeyError(f"[prepare_training_frame] 누락된 컬럼: {missing}")

        # 3) df_model 구성
        df_model = df[keep_cols].sort_values(by=cc.col_datetime).reset_index(drop=True)
        if verbose:
            print(
                f"✅ 기본 데이터 준비 완료 (행: {len(df_model):,}, 열: {len(df_model.columns)})"
            )

        # 4) valid_idx 계산 및 (옵션) 드랍
        valid_cols = feature_cols + [
            cc.col_lgbm_tmp_target_shift,
            cc.col_lgbm_tmp_weight,
        ]
        valid_idx = df_model[valid_cols].notna().all(axis=1)

        # (2025-09-08) (To-Do) logger 추가하고 debug 모드에서만 출력하도록 변경
        # if debug_nan:
        #     na = df_model[valid_cols].isna()
        #     print("\n[NaN 진단] 드랍 예정 행 수:", int(na.any(axis=1).sum()))
        #     print("[NaN 진단] 컬럼별 NaN 개수 (Top 20):")
        #     print(na.sum().sort_values(ascending=False).head(20))

        #     base_features = list(cc.lgbm_feature_columns)
        #     stat_features = [c for c in feature_cols if c not in base_features]
        #     print("\n[NaN 진단] 그룹별 NaN 유발 행 수:")
        #     print(" - base_features any NaN:", int(na[base_features].any(axis=1).sum()))
        #     print(" - stat_features any NaN:", int(na[stat_features].any(axis=1).sum()))
        #     print(" - target NaN:", int(na[cc.col_lgbm_tmp_target_shift].sum()))
        #     print(" - weight NaN:", int(na[cc.col_lgbm_tmp_weight].sum()))

        #     bad_ix = na.any(axis=1)
        #     bad_rows = df_model.index[bad_ix][:5]
        #     for i in bad_rows:
        #         bad_cols = na.columns[na.iloc[i]].tolist()
        #         print(f" - row {i} NaN in: {bad_cols[:10]}{' ...' if len(bad_cols)>10 else ''}")

        if drop_missing:
            before = len(df_model)
            df_model = df_model.loc[valid_idx].reset_index(drop=True)
            after = len(df_model)
            if verbose:
                removed = before - after
                pct = (removed / before * 100) if before else 0.0
                print(f"   전체 행: {before:,} → {after:,}")
                print(f"   제거된 행: {removed:,} ({pct:.1f}%)")

        # 5) ← 레거시 동일 위치: 드랍 이후, 고농도 추가 가중치 적용
        if apply_high_nox_weight:
            wcol = cc.col_lgbm_tmp_weight
            basis_col = (
                cc.col_lgbm_tmp_target_shift
                if high_nox_basis == "target"
                else cc.col_nox
            )
            lo, hi = (
                self.cfg.weight_high_nox_bound_lower,
                self.cfg.weight_high_nox_bound_upper,
            )

            mask = (
                (df_model[wcol] == self.cfg.weight_spike_neg)  # 기본 가중치(=1)인 곳만
                & (df_model[basis_col] > lo)
                & (df_model[basis_col] < hi)
            )
            if mask.any():
                df_model.loc[mask, wcol] = self.cfg.weight_high_nox
                if verbose:
                    print(
                        f"⚖️ 고농도 추가 가중치 적용(드랍 이후): {int(mask.sum()):,}행 → {self.cfg.weight_high_nox}"
                    )

        return df_model, feature_cols, valid_idx
