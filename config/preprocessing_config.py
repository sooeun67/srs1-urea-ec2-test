# ======================
# preprocessing_config.py
# ======================
"""
Preprocessing configuration for SKEP Urea Control System
"""
from __future__ import annotations

from dataclasses import dataclass, field

from collections.abc import Iterable
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from math import inf

import os
import logging

from utils.logger import LoggerConfig  # <- 오버라이드 대상
from config.column_config import ColumnConfig


@dataclass(frozen=True, slots=True)
class CommonPreprocessingConfig:
    """
    - 전처리 공통 설정(모델 무관)
    - 임계값 저장 방식: 실제_컬럼명 -> (lo, hi)
      * 정의는 변수키(col_*) 기준으로 하되, __post_init__에서 실제 컬럼명으로 해석/저장
    - 편의 접근:
      * pc.get_bounds("col_o2")      # 변수키
      * pc.get_bounds("br1_eo_o2_a") # 실제 컬럼명
      * pc.bound_o2                  # 변수키(property-like)
      * pc.bound__br1_eo_o2_a        # 실제 컬럼명(property-like)
    """

    # === 필수 ===
    column_config: ColumnConfig = field(default_factory=ColumnConfig)

    # 사업장 코드
    plant_code: str = "DEFAULT"

    # 실시간 집계 주기(초)
    resample_sec: int = 5

    # 최종 임계값: { 실제_컬럼명: (lo, hi) }
    global_threshold: Dict[str, Tuple[float, float]] = field(init=False)

    # 변수키(필수): ColumnConfig 속성명과 일치해야 함
    _numeric_var_keys: tuple[str, ...] = (
        "col_o2",
        "col_hz",
        "col_inner_temp",
        "col_outer_temp",
        "col_nox",
    )

    # 전사 기본 임계값 (변수키 기준)
    bounds_defaults_by_var: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "col_o2": (0.0, inf),
            "col_hz": (0.0, inf),
            "col_inner_temp": (0.0, inf),
            "col_outer_temp": (0.0, inf),
            "col_nox": (0.0, inf),
        }
    )

    # 플랜트별 오버라이드 (변수키 기준)
    bounds_by_plant: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=lambda: {
            "SRS1": {
                "col_hz": (20.0, 60.0),
                "col_o2": (0.001, 20.0),  # br1_eo_o2_a
                "col_inner_temp": (400.0, 1200.0),  # icf_ccs_fg_t_1
                "col_outer_temp": (400.0, 1200.0),  # icf_scs_fg_t_1
                "col_nox": (0.0, 200.0),  # icf_tms_nox_a
            },
            # 예) 다른 플랜트 추가 시 동일 패턴
            # "SRDD": {...},
        }
    )

    # --- 유틸: 변수키 -> 실제 컬럼명 ---
    def _colname_from_varkey(self, var_key: str) -> str:
        try:
            return getattr(self.column_config, var_key)
        except AttributeError as e:
            raise AttributeError(
                f"[ColumnConfig] '{var_key}' 속성이 없습니다. col_config를 확인하세요."
            ) from e

    # --- 유틸: (lo, hi) 검증 ---
    @staticmethod
    def _validate_bounds(tag: str, bounds: Tuple[float, float]) -> None:
        lo, hi = bounds
        if lo > hi:
            raise ValueError(f"Invalid bounds for {tag}: {bounds} (lo must be <= hi)")

    # --- 초기화: defaults + plant overrides 병합 → 실제 컬럼명 키로 저장 ---
    def __post_init__(self):
        code = (self.plant_code or "DEFAULT").upper()
        plant_overrides = self.bounds_by_plant.get(code, {})

        resolved: Dict[str, Tuple[float, float]] = {}

        for var_key in self._numeric_var_keys:
            default_bounds = self.bounds_defaults_by_var.get(var_key, (0.0, inf))
            plant_bounds = plant_overrides.get(var_key, default_bounds)

            self._validate_bounds(f"{code}.{var_key}", plant_bounds)

            # 변수키 → 실제 컬럼명 변환 후, 키를 정규화(strip)
            col_name = self._colname_from_varkey(var_key).strip()
            resolved[col_name] = plant_bounds

        object.__setattr__(self, "global_threshold", resolved)

    # --- 조회: 실제 컬럼명/변수키 모두 지원 ---
    def get_bounds(self, col_or_var: str) -> Tuple[float, float]:
        """
        col_or_var:
          - 실제 컬럼명(e.g., 'br1_eo_o2_a')
          - 변수키(e.g., 'col_o2')
        반환: (lo, hi) 또는 (0.0, inf)
        """
        cc = self.column_config
        key = col_or_var.strip()

        # 1) 실제 컬럼명 직접 조회
        if key in self.global_threshold:
            return self.global_threshold[key]

        # 2) 변수키(col_*)면 매핑 후 조회
        if hasattr(cc, key):
            col_name = getattr(cc, key)
            return self.global_threshold.get(str(col_name).strip(), (0.0, inf))

        # 3) fallback: 변수키를 훑어서 매칭(타이핑 오차나 공백 차이 방지)
        for varkey in self._numeric_var_keys:
            if hasattr(cc, varkey):
                v = getattr(cc, varkey)
                if str(v).strip() == key:
                    return self.global_threshold.get(str(v).strip(), (0.0, inf))

        return (0.0, inf)

    # --- sugar: bound_* / bound__* 속성 접근 ---
    def __getattr__(self, name: str):
        # 1) bound__<실제컬럼명> 우선 처리
        if name.startswith("bound__"):
            col_name = name[len("bound__") :].strip()
            return self.get_bounds(col_name)

        # 2) bound_<키> → col_<키> 로 변환하여 조회
        if name.startswith("bound_"):
            key = name[len("bound_") :].strip()  # "o2" | "hz" | "inner_temp" | ...
            var_key = f"col_{key}"
            return self.get_bounds(var_key)

        raise AttributeError(f"{name} not found")


# -------------------------
# 파생 설정들
# -------------------------


# 1) 학습 공통 (모델 공통으로 공유)
@dataclass(frozen=True)
class TrainPreprocessingConfig(CommonPreprocessingConfig):
    """
    학습용 공통 전처리 설정(모든 모델 공용)
    - eq_*         : EQ_Status 기반 행/구간 필터링 정책 (동일 클래스 내에서 직접 관리)
    - exclude_*    : 비정상 가동 구간 제외 정책
    - interpolate_*: 결측 보간 정책
    """

    # === EQ_Status 필터 ===
    # 상태==1 이후 추가 마스킹 시간(초)
    eq_shift_sec: int = 30
    # 모든 타깃 컬럼이 NaN인 연속 구간 제거 임계(초)
    eq_min_nan_block_sec: int = 600

    # === 비정상 가동 구간 제외 ===
    # INCINERATORSTATUS(또는 col_incinerator_status) 값이 아래 값과 같을 때,
    # 해당 시각의 앞뒤 window(분) 구간을 학습 데이터에서 제외합니다.
    exclude_status_value: int = 1  # 비정상 코드 값(기본 1)
    exclude_window_min: int = 40  # 제외 윈도우(분)

    # === 선형 보간 ===
    # 선형보간 한도
    # - (주의) pandas에 앞/뒤 최대 30초로 제한하는 것과 같은 기능은 없음
    # - 코드에서 별도로 gap을 계산해서 마스크 처리해야 함
    interpolate_limit_sec: int = 30
    # 선형보간 방법 (pandas 기준)
    # - "linear" : 값 자체를 직선 보간
    # - "time" : datetime index 기준 시간 간격을 고려해서 보간 (불규칙 샘플링 시 더 자연스러움)
    interpolate_method: Literal["linear", "time"] = "time"


# 2) 추론 공통 (모델 공통으로 공유)
@dataclass(frozen=True)
class InferPreprocessingConfig(CommonPreprocessingConfig):

    # ffill 한도(초)
    ffill_limit_sec: int = 20


# 3) GP 전처리


# === Train, GP ===
@dataclass(frozen=True)
class GPTrainPreprocessingConfig(TrainPreprocessingConfig):

    # === 시간 범위 설정 ===
    start_time: Optional[str] = "2025-08-12 00:00:00"
    end_time: Optional[str] = "2025-08-12 23:59:59"

    # === 학습 데이터 정책 ===
    min_samples: int = 8
    sample_size: Optional[int] = 1000  # 샘플링 개수
    random_state: int = 42
    dropna_required: bool = True  # 결측 제거
    dedup: bool = True  # 중복제거 (True: 활성화, False: 비활성화)

    # === 저장 설정 ===
    model_dir: str = "trained_models"
    model_basename: str = "gp_model"
    plant_code: Optional[str] = None

    # === 로깅 설정(오버라이드 지점) ===
    logger_cfg: LoggerConfig = field(
        default_factory=lambda: LoggerConfig(
            name="Preprocessing",
            level=logging.DEBUG,
            # 나머지는 logger.py 기본값(fmt/datefmt/propagate/refresh_handlers/use_stdout)을 그대로 상속
        )
    )

    @property
    def model_filename(self) -> str:
        base = self.model_basename
        if self.plant_code:
            base = f"{base}_{self.plant_code}"
        return f"{base}.joblib"

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, self.model_filename)


# === Infer, GP ===
# @dataclass(frozen=True)
# class GPInferPreprocessingConfig(InferPreprocessingConfig):

#     # ...
#    pass


# LGBM 요약통계량 column 규칙
def make_summary_feature_names(
    source_cols: Sequence[str],
    windows_summary_sec: Iterable[int],
    windows_rate_sec: Iterable[int],
) -> List[str]:
    """
    윈도우 설정과 원본 컬럼 목록으로 '실행 없이' 생성될 요약통계/모멘텀 컬럼명을 결정론적으로 생성.
    - 통계: {col}_mean_{s}s, {col}_std_{s}s
    - 모멘텀: {col}_momentum_max_up_{s}s, {col}_momentum_max_down_{s}s
    """
    names: List[str] = []
    for col in source_cols:
        for s in windows_summary_sec:
            names += [f"{col}_mean_{s}s", f"{col}_std_{s}s"]
        for s in windows_rate_sec:
            names += [
                f"{col}_momentum_max_up_{s}s",
                f"{col}_momentum_max_down_{s}s",
            ]
    return names


# 4) LGBM 전처리
@dataclass(frozen=True)
class _LGBMWindowsMixin:  # ← Common 상속 안 함!
    """LGBM용 윈도우 설정(공통 혼합용)"""

    # === 요약통계량 window(초) 설정 ===

    # window(초) 설정 (1) - 평균/표준편차
    windows_summary_sec: Iterable[int] = (60, 180, 300, 600)

    # window(초) 설정 (2) - 변화율(rate per sec) 최소/최대
    windows_rate_sec: Iterable[int] = (15, 30, 60)

    # === 요약통계량 column명 ===

    # ✅ 요약통계/모멘텀 생성 후 "실제 생성된 컬럼명"을 보관
    # - `generate_interval_summary_features_time`이 생성할 새로운 column들
    # - 생성자에서 받지 않음(자동 생성 전용)
    summary_feature_columns: List[str] = field(init=False, default_factory=list)

    # 자동 생성
    def __post_init__(self):
        # 상위 __post_init__ (Common→Train/Infer) 먼저
        try:
            super().__post_init__()
        except AttributeError:
            pass

        # 반드시 ColumnConfig에서 지정된 col_*만 사용 (fallback 없음)
        # - ❗ fallback 없음: 반드시 ColumnConfig.set_lgbm_feature_columns가 지정돼 있어야 함
        source_cols: Sequence[str] = (
            self.column_config.lgbm_feature_columns
        )  # 비어있으면 위에서 ValueError

        names = make_summary_feature_names(
            source_cols=source_cols,
            windows_summary_sec=self.windows_summary_sec,
            windows_rate_sec=self.windows_rate_sec,
        )
        object.__setattr__(self, "summary_feature_columns", names)


# === Train, LGBM ===
# - 공통 설정을 포함하면서 학습 배치 파이프라인에 맞는 필드 추가
@dataclass(frozen=True)
class LGBMTrainPreprocessingConfig(_LGBMWindowsMixin, TrainPreprocessingConfig):

    # === 학습 샘플 가중치 설정 ===

    # 학습 샘플 가중치 설정 (1) - Spike 구간
    weight_spike_delta_sec: int = 60  # 예측 시점 (초) -> 얼마 뒤 NOx를 예측할 것인지?
    weight_spike_step_sec: int = 30  # 연속 판정 간격 (초)
    weight_spike_thr_low: float = 25.0  # Spike 진입 전 저농도 기준
    weight_spike_thr_high: float = 40.0  # Spike 진입 고농도 기준
    weight_spike_lookback_sec: int = 300  # 고농도 진입 직전 window 길이 (초)
    weight_spike_pos: float = 10.0  # spike label=1 가중치
    weight_spike_neg: float = 1.0  # spike label=0 기본 가중치

    # 학습 샘플 가중치 설정 (2) - Spike가 아니고 NOx 고농도 구간
    weight_high_nox_bound_lower: float = 35.0
    weight_high_nox_bound_upper: float = 42.5
    weight_high_nox: float = 3.0


# === Infer, LGBM ===
# - 공통 설정을 포함하면서 실시간 5초 주기 추론에 맞는 필드 추가
@dataclass(frozen=True)
class LGBMInferPreprocessingConfig(_LGBMWindowsMixin, InferPreprocessingConfig):
    # ...
    # strict_schema: bool = True
    pass
