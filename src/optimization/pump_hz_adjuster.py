# ======================
# pump_hz_adjuster.py
# ======================



from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from config.column_config import ColumnConfig
from config.model_config import LGBMModelConfig
from config.rule_config import RuleConfig
from config.optimization_config import OptimizationConfig
from src.models.lgbm import LGBMNOxModel


@dataclass
class LGBMPumpHzAdjuster:
    """
    LGBM 기반 NOx 예측 + Hz 보정(룰) — 서빙용 최소 버전
    - 이미 피처가 들어있는 DataFrame을 받아서:
        1) NOx 예측 컬럼 추가
        2) LGBM 소프트 룰로 Hz 보정 컬럼 추가
    """
    column_config: ColumnConfig
    model_config: LGBMModelConfig
    rule_config: RuleConfig
    optimization_config: OptimizationConfig

    model: Optional[LGBMNOxModel] = field(default=None, repr=False)

    # --------------------------
    # 내부: 모델 로더 (글로벌/웜 재사용)
    # --------------------------
    def _ensure_model(self) -> LGBMNOxModel:
        if self.model is None:
            self.model = LGBMNOxModel(
                column_config=self.column_config,
                model_config=self.model_config
            ).load()
        return self.model

    # --------------------------
    # 1) 예측
    # --------------------------
    def predict_nox(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: Optional[Sequence[str]] = None,
        allow_nan: bool = True,
    ) -> pd.DataFrame:
        """
        입력 df(이미 피처 포함)에 대해 NOx 예측값을 cc.col_lgbm_db_pred_nox 에 기록.
        """
        cc = self.column_config
        mc = self.model_config
        model = self._ensure_model()

        feats = list(feature_cols or mc.lgbm_feature_columns_all)
        missing = [c for c in feats if c not in df.columns]
        if missing:
            raise KeyError(f"[predict_nox] 누락된 피처 컬럼: {missing}")

        out = df.copy()
        out[cc.col_lgbm_db_pred_nox] = model.predict_df(out, feature_cols=feats, allow_nan=allow_nan)
        return out

    # --------------------------
    # 2) 룰 적용 (SOFT 규칙)
    # --------------------------
    def apply_lgbm_soft_rule(
        self,
        df_pred: pd.DataFrame,
        *,
        return_flags: bool = False,
    ) -> pd.DataFrame:
        """
        레거시와 동일한 부등호:
          HARD   : y_pred > thr_nox
          MEDIUM : HARD & (hz_full < thr_hz)
          SOFT   : MEDIUM & (inner > thr_in?) &/or (outer > thr_out?)
                   - 내부/외부 중 '하나만' 있어도 그 하나 기준 적용
                   - 둘 다 있으면 AND
                   - 둘 다 없으면 SOFT 비활성화
        결과: cc.col_lgbm_db_hz_lgbm_adj = np.where(SOFT, hz_max, hz_full)
        """
        cc, rc, oc = self.column_config, self.rule_config, self.optimization_config

        pred_col    = cc.col_lgbm_db_pred_nox
        hz_full_col = cc.col_hz_full_rule
        inner_col   = getattr(cc, "col_inner_temp", None)
        outer_col   = getattr(cc, "col_outer_temp", None)

        need = [pred_col, hz_full_col]
        missing = [c for c in need if c not in df_pred.columns]
        if missing:
            raise KeyError(f"[apply_lgbm_soft_rule] 필수 컬럼 누락: {missing}")

        thr_nox   = rc.lgbm_cutoff_nox_pred_lower
        thr_hz    = rc.lgbm_cutoff_hz_suggest_upper
        thr_t_in  = rc.lgbm_cutoff_inner_temp_lower
        thr_t_out = rc.lgbm_cutoff_outer_temp_lower

        hz_max = float(oc.maximum_hz)

        out = df_pred.copy()

        # HARD / MEDIUM
        cond_hard   = out[pred_col] > thr_nox
        cond_medium = cond_hard & (out[hz_full_col] < thr_hz)

        # SOFT: 하나만 있으면 그 하나로, 둘 다 있으면 AND, 둘 다 없으면 비활성화
        cond_soft = cond_medium.copy()
        has_any = False
        if inner_col and inner_col in out.columns:
            cond_soft &= (out[inner_col] > thr_t_in)
            has_any = True
        if outer_col and outer_col in out.columns:
            cond_soft &= (out[outer_col] > thr_t_out)
            has_any = True
        if not has_any:
            cond_soft = pd.Series(False, index=out.index)

        out[cc.col_lgbm_db_hz_lgbm_adj] = np.where(cond_soft, hz_max, out[hz_full_col].astype(float))

        if return_flags:
            out = out.assign(_rule_hard=cond_hard.values,
                             _rule_medium=cond_medium.values,
                             _rule_soft=cond_soft.values)
        return out

    # --------------------------
    # (옵션) 예측 + 룰 적용 한 번에
    # --------------------------
    def predict_and_adjust(
        self,
        df_with_features: pd.DataFrame,
        *,
        feature_cols: Optional[Sequence[str]] = None,
        allow_nan: bool = True,
        return_flags: bool = False,
    ) -> pd.DataFrame:
        """
        이미 피처가 포함된 DF를 받아서:
          1) NOx 예측 → 2) 룰 적용 까지 한번에.
        """
        df_pred = self.predict_nox(df_with_features, feature_cols=feature_cols, allow_nan=allow_nan)
        return self.apply_lgbm_soft_rule(df_pred, return_flags=return_flags)
