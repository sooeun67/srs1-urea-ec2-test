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
    bounds_o2_table: List[Tuple[float, Tuple[Optional[float], Optional[float]]]] = (
        field(
            default_factory=lambda: [
                (4.0, (49.0, None)),
                (5.0, (47.0, None)),
                (6.0, (45.0, None)),
                (7.0, (43.0, 49.0)),
                (8.0, (40.0, 46.0)),
                (9.0, (None, 44.0)),
                (10.0, (None, 42.0)),
                (11.0, (None, 40.0)),
                (12.0, (None, 39.0)),
            ]
        )
    )
    # 표의 마지막 경계 초과 시 적용
    bounds_o2_else: Tuple[Optional[float], Optional[float]] = (None, None)

    # 동적 경계 임계값
    dyn_temp_in_low: float = 850.0
    dyn_temp_in_high: float = 1200.0
    dyn_temp_out_low: float = 850.0
    dyn_temp_out_high: float = 1000.0
    dyn_nox_low: float = 7.0
    dyn_nox_high: float = 40.0
    dyn_o2_low: float = 5.0
    dyn_o2_high: float = 10.0

    # 파이프라인 옵션
    dropna_required: bool = True  # 입력필수 컬럼 결측 제거
    keep_outer_merge: bool = True  # 원본 시각 전체와 외부 병합
    progress: bool = False  # tqdm 진행 표시

    # ---------------------------------
    # LGBM 기반 보정 규칙 Cutoffs (cond_hard / cond_medium / cond_soft 에서 사용)
    # ---------------------------------

    # (2025-09-08) 수정 필요 사항 메모
    # (As-Is) AND 조건을 만족하면, ~/config/optimization_config.py 내의 `maximum_hz`로 고정
    # (To-Do) 추가 Rule 필요

    # (HARD) 조건: 예측 NOx가 하한을 초과할 때 트리거
    #   cond_hard   = (y_pred > lgbm_cutoff_nox_pred_lower)
    lgbm_cutoff_nox_pred_lower: float = 30.0

    # (MEDIUM) 조건: HARD 조건 + '보정 전(Full Rule 적용값) 추천 Hz'가 상한 '미만'일 때
    #   cond_medium = cond_hard & (suggested_pump_hz_full < lgbm_cutoff_hz_suggest_upper)
    lgbm_cutoff_hz_suggest_upper: float = 40.0

    # (SOFT) 조건: MEDIUM 조건 + 내부/외부 온도가 하한을 '초과'할 때
    #   cond_soft   = cond_medium
    #                 & (inner_temp > lgbm_cutoff_inner_temp_lower)
    #                 & (outer_temp > lgbm_cutoff_outer_temp_lower)
    lgbm_cutoff_inner_temp_lower: float = 850.0
    lgbm_cutoff_outer_temp_lower: float = 850.0
