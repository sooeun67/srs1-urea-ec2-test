# ======================
# model_config.py
# ======================
"""
Model configuration for SKEP Urea Control System
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import os

@dataclass
class ModelConfig:
    """Gaussian Process 모델/학습/저장 설정"""

    # === 시간 범위 설정 ===
    start_time: Optional[str] = '2025-08-12 00:00:00'   # 학습/추론 시작 시각
    end_time  : Optional[str] = '2025-08-12 23:59:59'   # 학습/추론 종료 시각

    # === GP 모델 설정 ===
    normalize_y: bool = True
    alpha: float = 1e-6
    n_restarts_optimizer: int = 5
    random_state: int = 42

    # === 커널 파라미터 ===
    # ConstantKernel
    constant_value: float = 1.0
    constant_bounds: Tuple[float, float] = (1e-3, 1e3)

    # MaternKernel
    matern_nu: float = 2.5
    matern_length_scale_init: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # [Hz, O2, Temp]

    # WhiteKernel
    white_noise_level: float = 1.0
    white_noise_bounds: Tuple[float, float] = (1e-6, 1e2)

    # === 학습 데이터 정책 ===
    min_samples: int = 8
    sample_size: Optional[int] = 1000   # None이면 전체 사용
    dropna_required: bool = True
    dedup: bool = True                  # 중복 샘플 제거(피처+타깃 기준)

    # === 저장 설정 ===
    model_dir: str = "trained_models"
    model_basename: str = "gp_model"
    plant_code: Optional[str] = None    # 예: "SRS1", "SRDD"

    @property
    def model_filename(self) -> str:
        base = self.model_basename
        if self.plant_code:
            base = f"{base}_{self.plant_code}"
        return f"{base}.joblib"

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, self.model_filename)
