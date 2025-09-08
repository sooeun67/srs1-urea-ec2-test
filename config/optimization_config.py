# ======================
# optimization_config.py
# ======================
"""
Optimization configuration for SKEP Urea Control System
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class OptimizationConfig:
    """최적화/탐색 설정"""

    # 기본 최적화 파라미터
    target_nox: float = 10.0
    p_feasible: float = 0.90
    n_candidates: int | None = None
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

    @property
    def pump_bounds(self) -> Tuple[float, float]:
        """펌프 Hz 범위 (min, max)"""
        return (float(self.minimum_hz), float(self.maximum_hz))

    def get_z_value(self, p_feasible: float | None = None) -> float:
        """
        안전확률 p_feasible를 가장 가까운(이상) 키로 매핑하여 z 반환.
        - 기본 전략: p 이상인 키 중 최솟값 선택
        - 매칭 없으면 0.0
        """
        p = self.p_feasible if p_feasible is None else float(p_feasible)
        keys = sorted(self.safety_probability_mapping.keys(), reverse=True)
        for thr in keys:
            if p >= thr:
                return float(self.safety_probability_mapping[thr])
        return 0.0
