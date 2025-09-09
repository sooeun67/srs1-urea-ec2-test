# ======================
# optimization_config.py
# ======================
"""
Optimization configuration for SKEP Urea Control System
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import logging

from utils.logger import LoggerConfig  # ← 로깅 오버라이드 지점


@dataclass
class OptimizationConfig:
    """최적화/탐색 설정"""

    # 기본 최적화 파라미터
    target_nox: float = 10.0
    p_feasible: float = 0.90
    n_candidates: Optional[int] = None
    round_to_int: bool = True

    # 펌프 Hz 전역 범위
    minimum_hz: float = 38.0
    maximum_hz: float = 54.0

    # ✅ 입력 결측시 사용할 Fallback Hz (None이면 maximum_hz 사용)
    fallback_hz: Optional[float] = 43.0

    # 안전 확률 → z 값 매핑
    safety_probability_mapping: Dict[float, float] = field(
        default_factory=lambda: {
            0.95: 1.60,
            0.90: 1.2816,
            0.85: 1.036,
            0.80: 0.8416,
        }
    )

    # === 로깅 설정(오버라이드 지점) ===
    # 필요 시 model_config.py처럼 여기서 name/level만 바꿔서 사용
    logger_cfg: LoggerConfig = field(
        default_factory=lambda: LoggerConfig(
            name="PumpOptimizer",
            level=logging.DEBUG,  # 필요에 따라 INFO/DEBUG 등으로 조정
            # fmt/datefmt/propagate/refresh_handlers/use_stdout 는 logger.py 기본값 사용
        )
    )

    @property
    def pump_bounds(self) -> Tuple[float, float]:
        """펌프 Hz 범위 (min, max)"""
        return (float(self.minimum_hz), float(self.maximum_hz))

    def get_z_value(self, p_feasible: Optional[float] = None) -> float:
        """
        안전확률 p_feasible를 가장 가까운(이상) 키로 매핑하여 z 반환.
        - 기본: p 이상인 키 중 최솟값 선택
        - 매칭 없으면 0.0
        """
        p = self.p_feasible if p_feasible is None else float(p_feasible)
        keys = sorted(self.safety_probability_mapping.keys(), reverse=True)
        for thr in keys:
            if p >= thr:
                return float(self.safety_probability_mapping[thr])
        return 0.0
