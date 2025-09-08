# ======================
# rule_config.py
# ======================
"""
Rule configuration for pump Hz optimization
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class RuleConfig:
    """펌프 Hz 규칙/경계 설정"""

    # O2 기반 정적 경계표: (O2 상한, (lo_opt, hi_opt))
    # None은 해당 방향으로 전역(min/max)을 의미
    bounds_o2_table: List[Tuple[float, Tuple[Optional[float], Optional[float]]]] = field(
        default_factory=lambda: [
            (4.0 , (49.0, None)),
            (5.0 , (47.0, None)),
            (6.0 , (45.0, None)),
            (7.0 , (43.0, 49.0)),
            (8.0 , (40.0, 46.0)),
            (9.0 , (None, 44.0)),
            (10.0, (None, 42.0)),
            (11.0, (None, 40.0)),
            (12.0, (None, 39.0)),
        ]
    )
    # 표의 마지막 경계 초과 시 적용
    bounds_o2_else: Tuple[Optional[float], Optional[float]] = (None, None)

    # 동적 경계 임계값
    dyn_temp_in_low  : float = 850.0
    dyn_temp_in_high : float = 1200.0
    dyn_temp_out_low : float = 850.0
    dyn_temp_out_high: float = 1000.0
    dyn_nox_low      : float = 7.0
    dyn_nox_high     : float = 40.0
    dyn_o2_low       : float = 5.0
    dyn_o2_high      : float = 10.0

    # 파이프라인 옵션
    dropna_required: bool = True          # 입력필수 컬럼 결측 제거
    keep_outer_merge: bool = True         # 원본 시각 전체와 외부 병합
    progress: bool = False                # tqdm 진행 표시
