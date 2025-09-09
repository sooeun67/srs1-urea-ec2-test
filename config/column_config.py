# ======================
# column_config.py
# ======================
"""
Column configuration for SKEP Urea Control System

- dataclass(frozen=True): 생성 후 불변 보장
- plant_code 프리셋은 __post_init__에서 object.__setattr__로 1회 적용
- 런타임 오버라이드는 dataclasses.replace 기반 with_overrides() 사용
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields, replace
from typing import List, Tuple, Dict, Any, Optional
from itertools import zip_longest  # 파일 상단 import에 추가

# 사이트별 기본 프리셋
PLANT_CODE_PRESETS: Dict[str, Dict[str, Any]] = {
    # SRS1: 내부온도 사용
    "SRS1": {
        "col_temp": "icf_ccs_fg_t_1",  # = col_inner_temp
        "cols_temp": ["icf_ccs_fg_t_1", "icf_scs_fg_t_1"],
        "set_lgbm_feature_columns": [
            "col_nox",
            "col_o2",
            "col_inner_temp",
            "col_outer_temp",
        ],
    },
    # SRDD: 출구온도 사용(내부온도 센서 미존재 가정)
    "SRDD": {
        "col_temp": "icf_scs_fg_t_1",  # = col_outer_temp
        "cols_temp": ["icf_scs_fg_t_1"],
        "set_lgbm_feature_columns": ["col_nox", "col_o2", "col_outer_temp"],
    },
}


@dataclass(frozen=True)
class ColumnConfig:
    """
    데이터 컬럼 설정 (모델/추천/메타 컬럼 일원화)

    Parameters
    ----------
    plant_code : Optional[str]
        사이트 프리셋 키. 예: "SRS1", "SRDD".
        None이면 프리셋 적용 없음.
    """

    # ----- 프리셋 제어 메타 -----
    plant_code: Optional[str] = None  # 위치 인자 ColumnConfig("SRS1")로도 전달 가능

    # ----- 원본 컬럼 -----
    col_datetime: str = "_time_gateway"
    col_o2: str = "br1_eo_o2_a"  # 보일러 출구 산소 농도
    col_hz: str = "snr_pmp_uw_s_1"  # 실제 요소수 펌프 Hz
    col_inner_temp: str = "icf_ccs_fg_t_1"  # 소각로 내부 온도 (없을 수 있음)
    col_outer_temp: str = "icf_scs_fg_t_1"  # 소각로 출구 온도
    col_nox: str = "icf_tms_nox_a"  # NOx (보정 전/후 무관, 입력 기준)
    col_ai: str = "acc_snr_ai_1a"  # 요소수 AI 모드 여부
    col_act_status: str = "act_status"
    col_inc_status: str = "incineratorstatus"

    # ✅ 모델 입력용 '대표 온도' (사이트별로 inner/outer 중 택1)
    col_temp: str = "icf_ccs_fg_t_1"
    # 온도 변수(가변 길이) → tuple로 보관
    cols_temp: List[str] = field(default_factory=list)
    # cols_temp: Tuple[str, ...] = field(default_factory=tuple)

    # --- EQ Status 필터링 관련 컬럼 세트 ---
    # 원본(TMS) 측정치 컬럼
    cols_icf_tms: List[str] = field(
        default_factory=lambda: ["icf_tms_nox_a", "icf_tms_o2_a"]
    )
    # 보정/가공된 값 컬럼
    cols_tms_value: List[str] = field(
        default_factory=lambda: ["nox_value", "o2b_value"]
    )
    # 해당 값들의 EQ Status 컬럼
    cols_tms_eq_status: List[str] = field(
        default_factory=lambda: ["nox_eq_status", "o2b_eq_status"]
    )
    col_real_row: str = "is_real_row"
    col_eq_status_filtered: str = "is_eq_status_filtered"
    col_inc_status_filtered: str = "is_inc_status_filtered"
    col_glob_threshold_filtered: str = "is_glob_threshold_filtered"

    # ----- 모델 피처/타깃 -----
    col_pred_mean: str = "pred_nox_mean"
    col_pred_ucb: str = "pred_nox_ucb"
    col_target_nox: str = "target_nox"

    # ----- 메타 정보 -----
    col_safety_gap: str = "safety_gap_to_target"
    col_p_feasible: str = "p_feasible"
    col_round_flag: str = "round_to_int"
    col_n_candidates: str = "n_candidates"
    col_grid_min: str = "grid_min"
    col_grid_max: str = "grid_max"
    col_grid_size: str = "grid_size"

    # 성대 act_snr_pmp_uw_s
    # ----- 추천 결과 컬럼 -----
    col_hz_out: str = "suggested_pump_hz_temp"
    col_hz_raw_out: str = "act_snr_pmp_bo_1"  # GP 결과
    col_hz_init_rule: str = "act_snr_pmp_bo_2"  # O2 규칙 적용
    col_hz_full_rule: str = "act_snr_pmp_bo_3"  # O2 + 동적 규칙

    @property
    def target_column(self) -> str:
        return self.col_nox

    # (09-04, 믿음, 변경)
    # 기존:feature_columns -> 수정:gp_feature_columns
    # @property
    # def feature_columns(self) -> List[str]:
    #     """모델 입력 피처 컬럼 순서: [Hz, O2, Temp]"""
    #     return [self.col_hz, self.col_o2, self.col_temp]
    @property
    def gp_feature_columns(self) -> List[str]:
        """Gaussian Process 모델 입력 피처 컬럼 순서: [Hz, O2, Temp]"""
        return [self.col_hz, self.col_o2, self.col_temp]

    # === LightGBM 모형에 사용할 DB 원본 feature 목록 ===

    # 어떤 컬럼을 feature로 쓸지 지정
    set_lgbm_feature_columns: List[str] = field(default_factory=list)

    @property
    def lgbm_feature_columns(self) -> List[str]:
        """LightGBM 모델 입력 피처 컬럼 반환 (col_ 로 시작하는 필드만)"""
        # col_ 로 시작하는 필드만 후보
        all_cols = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name.startswith("col_")
        }

        # 지정된 feature 이름이 올바른지 검증
        unknown = [c for c in self.set_lgbm_feature_columns if c not in all_cols]
        if unknown:
            raise KeyError(
                f"Unknown feature(s): {unknown}. " f"Available: {list(all_cols.keys())}"
            )

        return [all_cols[name] for name in self.set_lgbm_feature_columns]

    # === LightGBM 코드로 생성하는 column ===

    # (1) DB 저장 X
    col_lgbm_tmp_weight: str = "lgbm_train_weight"  # train 시 sample 가중치
    col_lgbm_tmp_target_shift: str = (
        "lgbm_train_target_shift"  # train 시 미래값(shifted ground truth)
    )
    col_lgbm_tmp_is_spike: str = (
        "lgbm_train_is_spike"  # 스파이크 전이구간(시프트 반영) 내 여부(True/False)
    )
    col_lgbm_tmp_flag_interval_hit: str = (
        "lgbm_train_flag_interval_hit"  # 전이구간 히트 여부 플래그
    )
    col_lgbm_tmp_has_target: str = (
        "lgbm_train_has_target"  # 해당 시점에 유효한 타깃 존재 여부
    )

    # (참고) LGBM 요약통계량 column
    # - naming rule: ~/config/preprocessing_config.py에 있음
    # - 실제 column명은 원본 column에 따라 달라짐

    # (2) DB 저장 O
    col_lgbm_db_pred_nox: str = (
        "snr_nox_pred"  # 현재 시점에서 1분(or 2.5분?) 뒤 미래 NOx 예측값
    )
    col_lgbm_db_hz_lgbm_adj: str = (
        "act_snr_pmp_bo_4"  # 미래 예측값(+추가 Rule)으로 보정된 Hz 추천값
    )

    # --------- Lifecycle hooks / factories ---------
    def __post_init__(self):
        """
        frozen=True 환경에서 plant_code 프리셋을 1회 주입.
        - object.__setattr__ 사용(허용된 패턴)
        """
        if self.plant_code:
            key = str(self.plant_code).upper()
            preset = PLANT_CODE_PRESETS.get(key)
            if preset:
                for k, v in preset.items():
                    # 필요 시 불변화하려면 여기서 tuple 변환 가능:
                    # if k in ("cols_temp", "set_lgbm_feature_columns") and isinstance(v, list):
                    #     v = tuple(v)
                    object.__setattr__(self, k, v)

    def with_overrides(self, **overrides: Any) -> "ColumnConfig":
        """
        불변 인스턴스에 오버라이드 적용한 새 인스턴스 반환.
        예) cc2 = cc.with_overrides(col_temp="icf_scs_fg_t_1")
        """
        return replace(self, **overrides)

    @classmethod
    def from_plant_code(cls, plant_code: str, **overrides: Any) -> "ColumnConfig":
        """
        사이트 프리셋 + 임의 오버라이드를 한 번에 생성.
        예) ColumnConfig.from_plant_code("SRDD", set_lgbm_feature_columns=[...])
        """
        base = cls(plant_code=plant_code)
        return base.with_overrides(**overrides) if overrides else base

    @classmethod
    def from_args(cls, args: Any) -> "ColumnConfig":
        """
        argparse.Namespace 호환 팩토리.
        - args.plant_code, args.col_temp 등 전달 시 오버라이드
        """
        plant_code = getattr(args, "plant_code", None)
        cc = cls(plant_code=plant_code)  # 프리셋 적용
        # 개별 오버라이드(존재할 때만)
        kv = {}
        for name in [
            "col_temp",
            "col_hz",
            "col_o2",
            "col_inner_temp",
            "col_outer_temp",
            "col_nox",
            "col_datetime",
            # ↓ 추가: 리스트 인자도 허용
            "cols_icf_tms",
            "cols_tms_value",
            "cols_tms_eq_status",
        ]:
            if hasattr(args, name) and getattr(args, name) is not None:
                kv[name] = getattr(args, name)
        if (
            hasattr(args, "lgbm_features")
            and getattr(args, "lgbm_features") is not None
        ):
            # 콤마 구분 문자열 또는 리스트 모두 허용
            lf = getattr(args, "lgbm_features")
            kv["set_lgbm_feature_columns"] = (
                [s.strip() for s in lf.split(",")] if isinstance(lf, str) else list(lf)
            )
        return cc.with_overrides(**kv) if kv else cc

    @property
    def eq_map(self) -> Dict[str, List[str]]:
        """
        EQ_Status 컬럼명 → [보정값컬럼, 원본(TMS)컬럼] 매핑
        예: {"nox_eq_status": ["nox_value", "icf_tms_nox_a"], ...}
        """
        a, b, c = self.cols_tms_eq_status, self.cols_tms_value, self.cols_icf_tms
        if not (len(a) == len(b) == len(c)):
            raise ValueError(
                f"[ColumnConfig.eq_map] 길이 불일치: "
                f"eq_status={len(a)}, tms_value={len(b)}, icf_tms={len(c)}"
            )
        return {eq: [val, icf] for eq, val, icf in zip(a, b, c)}


# def _prepare_cc_kwargs(
#     plant_code: Optional[str],
#     overrides: Optional[Dict[str, Any]] = None
# ) -> Dict[str, Any]:
#     """프리셋 + 오버라이드 병합 후 안전한 kwargs 생성 (list→tuple 변환 포함)"""
#     merged: Dict[str, Any] = {}

#     if plant_code:
#         preset = PLANT_CODE_PRESETS.get(str(plant_code).upper())
#         if preset:
#             merged.update(preset)

#     if overrides:
#         merged.update({k: v for k, v in overrides.items() if v is not None})

#     # list → tuple (frozen dataclass 안전성)
#     if "cols_temp" in merged and isinstance(merged["cols_temp"], list):
#         merged["cols_temp"] = tuple(merged["cols_temp"])
#     if "set_lgbm_feature_columns" in merged and isinstance(merged["set_lgbm_feature_columns"], list):
#         merged["set_lgbm_feature_columns"] = tuple(merged["set_lgbm_feature_columns"])

#     return merged
