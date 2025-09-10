# ======================
# model_config.py
# ======================
"""
Model configuration for SKEP Urea Control System
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple
import os
import logging

from utils.logger import LoggerConfig  # <- 오버라이드 대상
from config.column_config import ColumnConfig


@dataclass
class GPModelConfig:
    column_config: Optional[ColumnConfig] = None

    def __post_init__(self) -> None:
        self.cc = self.column_config or ColumnConfig()

    # === 시간 범위 설정 ===
    start_time: Optional[str] = "2025-08-12 00:00:00"
    end_time: Optional[str] = "2025-08-12 23:59:59"

    # === GP 모델 설정 ===
    normalize_y: bool = True
    alpha: float = 1e-6
    n_restarts_optimizer: int = 5
    random_state: int = 42

    # === 커널 파라미터 ===
    constant_value: float = 1.0
    constant_bounds: Tuple[float, float] = (1e-3, 1e3)
    matern_nu: float = 2.5
    matern_length_scale_init: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    white_noise_level: float = 1.0
    white_noise_bounds: Tuple[float, float] = (1e-6, 1e2)

    # === 학습 데이터 정책 ===
    min_samples: int = 8
    sample_size: Optional[int] = 1000  # 샘플링 개수
    dropna_required: bool = True  # 결측 제거
    dedup: bool = True  # 중복제거 (True: 활성화, False: 비활성화)

    # === 저장 설정 ===
    model_dir: str = "trained_models"
    model_basename: str = "gp_model"
    plant_code: Optional[str] = None

    # === 로깅 설정(오버라이드 지점) ===
    logger_cfg: LoggerConfig = field(
        default_factory=lambda: LoggerConfig(
            name="GaussianProcessNOxModel",
            level=logging.DEBUG,
            # 나머지는 logger.py 기본값(fmt/datefmt/propagate/refresh_handlers/use_stdout)을 그대로 상속
        )
    )

    # -----------------------------
    # Internal helpers (원래 있던 것만 유지)
    # -----------------------------
    def _training_required(self) -> List[str]:
        return list(
            dict.fromkeys(
                self.cc.gp_feature_columns
                + self.cc.lgbm_feature_columns
                + [self.cc.target_column]
            )
        )

    def _training_columns(self) -> List[str]:
        cols = [self.cc.col_datetime] + self._training_required()
        cols += [self.cc.col_ai, self.cc.col_act_status, self.cc.col_inc_status]
        cols += (
            self.cc.cols_temp
            + self.cc.cols_icf_tms
            + self.cc.cols_tms_value
            + self.cc.cols_tms_eq_status
        )
        return list(dict.fromkeys(cols))

    def _inference_required(self) -> List[str]:
        return list(
            dict.fromkeys(
                self.cc.gp_feature_columns
                + self.cc.lgbm_feature_columns
                + [self.cc.target_column]
            )
        )

    def _inference_columns(self) -> List[str]:
        cols = [self.cc.col_datetime] + self._inference_required()
        cols += self.cc.cols_temp
        return list(dict.fromkeys(cols))

    @property
    def model_filename(self) -> str:
        base = self.model_basename
        if self.plant_code:
            base = f"{base}_{self.plant_code}"
        return f"{base}.joblib"

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, self.model_filename)


@dataclass(frozen=False)  # [0910] 수정 (frozen=True) 여야 함
class LGBMModelConfig:

    # === 시간 범위 설정 ===
    start_time: Optional[str] = "2025-07-01 00:41:30"
    end_time: Optional[str] = "2025-08-11 23:20:00"

    # === LightGBM 기본 설정 ===
    lgbm_random_state: int = 42
    lgbm_n_estimators: int = 200
    lgbm_learning_rate: float | None = None
    lgbm_max_depth: int | None = None
    lgbm_reg_alpha: float | None = None  # L1
    lgbm_reg_lambda: float | None = None  # L2
    lgbm_objective: Literal[
        "regression",  # MSE
        "regression_l1",  # MAE
        "quantile",
    ] = "regression"
    # objective 의존 파라미터
    lgbm_quantile_alpha: float | None = None  # objective="quantile"일 때 (0,1)

    # === 모델 입력 컬럼 관리 ===
    # (1) 원본 컬럼명: ColumnConfig.lgbm_feature_columns 결과를 그대로 복사해 넣기
    lgbm_feature_columns_original: List[str] = field(default_factory=list)
    # (2) 전처리 요약통계/모멘텀 컬럼명: preprocessing에서 생성된 리스트를 넣기
    lgbm_feature_columns_summary: List[str] = field(default_factory=list)

    @property
    def lgbm_feature_columns_all(self) -> List[str]:
        """(1)+(2)를 순서대로 합치되, 중복은 제거(첫 등장 우선)."""
        return list(
            dict.fromkeys(
                self.lgbm_feature_columns_original + self.lgbm_feature_columns_summary
            )
        )

    # LightGBM 파라미터 dict로 변환
    def to_lgbm_params(self) -> dict:
        """
        Convert this config to a LightGBM parameter dictionary.

        - None 값은 dict에서 자동 제거
        - objective="quantile"일 때 alpha 필요
        """
        params = {
            "random_state": self.lgbm_random_state,
            "n_estimators": self.lgbm_n_estimators,
            "objective": self.lgbm_objective,
            "learning_rate": self.lgbm_learning_rate,
            "max_depth": self.lgbm_max_depth,
            "reg_alpha": self.lgbm_reg_alpha,
            "reg_lambda": self.lgbm_reg_lambda,
        }

        # quantile objective 보완
        if self.lgbm_objective == "quantile":
            if self.lgbm_quantile_alpha is None:
                raise ValueError(
                    "quantile objective requires lgbm_quantile_alpha to be set."
                )
            params["alpha"] = self.lgbm_quantile_alpha

        # None 값 제거
        return {k: v for k, v in params.items() if v is not None}

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

    # === 저장 경로들 ===

    model_path: str = "./artifacts/lgbm_model.joblib"  # joblib(sklearn) 저장용 (옵션)
    meta_path: str = "./artifacts/lgbm_meta.json"  # 메타정보 저장용

    # ✅ 네이티브 LightGBM Booster 경로 (없으면 자동으로 model_path 기반 *.txt 사용)
    native_model_path: Optional[str] = None

    # === 로깅 설정(오버라이드 지점) ===
    logger_cfg: LoggerConfig = field(
        default_factory=lambda: LoggerConfig(
            name="LGBMNOxModel",
            level=logging.DEBUG,
            # 나머지는 logger.py 기본값(fmt/datefmt/propagate/refresh_handlers/use_stdout)을 그대로 상속
        )
    )
