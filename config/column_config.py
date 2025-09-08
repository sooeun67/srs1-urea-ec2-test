# ======================
# column_config.py
# ======================
"""
Column configuration for SKEP Urea Control System
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class ColumnConfig:
    """데이터 컬럼 설정 (모델/추천/메타 컬럼 일원화)"""

    # 원본 컬럼
    col_datetime: str = "_time_gateway"
    col_o2: str = "br1_eo_o2_a"              # 보일러 출구 산소 농도
    col_hz: str = "snr_pmp_uw_s_1"           # 실제 요소수 펌프 Hz
    col_inner_temp: str = "icf_ccs_fg_t_1"   # 소각로 내부 온도 (없을 수 있음)
    col_outer_temp: str = "icf_scs_fg_t_1"   # 소각로 출구 온도
    col_nox: str = "icf_tms_nox_a"           # NOx (보정 전/후 무관, 입력 기준)
    col_ai: str = "acc_snr_ai_1a"            # 요소수 AI 모드 여부

    # ✅ 모델 입력용 '대표 온도' (사이트별로 inner/outer 중 택1)
    col_temp: str = "icf_ccs_fg_t_1"

    # 모델 피처/타깃
    col_pred_mean: str = "pred_nox_mean"
    col_pred_ucb: str = "pred_nox_ucb"
    col_target_nox: str = "target_nox"

    # 메타 정보
    col_safety_gap: str = "safety_gap_to_target"
    col_p_feasible: str = "p_feasible"
    col_round_flag: str = "round_to_int"
    col_n_candidates: str = "n_candidates"
    col_grid_min: str = "grid_min"
    col_grid_max: str = "grid_max"
    col_grid_size: str = "grid_size"

    # 추천 결과 컬럼
    col_hz_out: str = "suggested_pump_hz"
    col_hz_raw_out: str = "suggested_pump_hz_raw"     # GP 결과
    col_hz_init_rule: str = "suggested_pump_hz_o2"    # O2 규칙 적용
    col_hz_full_rule: str = "suggested_pump_hz_full"  # O2 + 동적 규칙

    @property
    def feature_columns(self) -> List[str]:
        """모델 입력 피처 컬럼 순서: [Hz, O2, Temp]"""
        return [self.col_hz, self.col_o2, self.col_temp]

    @property
    def target_column(self) -> str:
        return self.col_nox


# SRS1: 내부온도를 모델 입력으로 사용
cc_srs1 = ColumnConfig(
    col_temp="icf_ccs_fg_t_1",      # = col_inner_temp
    col_inner_temp="icf_ccs_fg_t_1",
    col_outer_temp="icf_scs_fg_t_1",
)

# SRDD: 내부온도 센서 없음 → 출구온도를 모델 입력으로 사용
cc_srdd = ColumnConfig(
    col_temp="icf_scs_fg_t_1",      # = col_outer_temp
    col_inner_temp="icf_ccs_fg_t_1",
    col_outer_temp="icf_scs_fg_t_1",
)
