# ======================
# pump_optimizer.py
# ======================
"""
Pump optimization for urea injection control (GP + UCB + Rules)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import os
import logging
import numpy as np
import pandas as pd

from config.column_config import ColumnConfig
from config.optimization_config import OptimizationConfig
from config.rule_config import RuleConfig
from src.models.gaussian_process import GaussianProcessNOxModel # [0910] 경로 수정
from utils.logger import get_logger


@dataclass
class PumpOptimizer:
    """요소수 펌프 Hz 최적화기 (GP-UCB + 정적/동적 경계 후처리)"""

    model: GaussianProcessNOxModel

    column_config: ColumnConfig = field(default_factory=ColumnConfig)
    opt_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    rule_config: RuleConfig = field(default_factory=RuleConfig)

    # 로거 (config 오버라이드 사용)
    logger: logging.Logger = field(init=False, repr=False)

    # __post_init__ 교체
    def __post_init__(self) -> None:
        # 오직 opt_config.logger_cfg만 사용
        cfg = getattr(self.opt_config, "logger_cfg", None)
        if cfg is None:
            raise RuntimeError(
                "opt_config.logger_cfg가 없습니다. OptimizationConfig에 logger_cfg를 설정하세요."
            )
        self.logger = get_logger(cfg)

    # ----------------------
    # 내부 유틸 (내장: 시작 로그 생략, 디버그만)
    # ----------------------
    def _create_candidate_grid(
        self, lo: float, hi: float, n_candidates: Optional[int], round_to_int: bool
    ) -> np.ndarray:
        self.logger.debug(
            f"그리드 생성 요청: bounds=({lo}, {hi}), n_candidates={n_candidates}, round_to_int={round_to_int}"
        )
        lo = float(lo)
        hi = float(hi)
        if lo > hi:
            raise ValueError(f"잘못된 bounds: ({lo}, {hi})")

        if n_candidates is None:
            if round_to_int:
                lo_i = int(np.ceil(lo))
                hi_i = int(np.floor(hi))
                if hi_i < lo_i:
                    raise ValueError(f"정수 그리드를 만들 수 없음: bounds=({lo}, {hi})")
                grid = np.arange(lo_i, hi_i + 1, 1, dtype=float)
            else:
                default_n = max(int(np.ceil((hi - lo) * 10)) + 1, 50)
                grid = np.linspace(lo, hi, default_n, dtype=float)
        else:
            grid = np.linspace(lo, hi, int(n_candidates), dtype=float)
            if round_to_int:
                grid_i = np.rint(grid).astype(int)
                grid_i = np.clip(grid_i, int(np.ceil(lo)), int(np.floor(hi)))
                grid_i = np.unique(grid_i)
                if grid_i.size == 0:
                    raise ValueError(
                        f"정수 그리드를 만들 수 없음: bounds=({lo}, {hi}), n={n_candidates}"
                    )
                grid = grid_i.astype(float)

        self.logger.debug(f"그리드 생성 완료:")
        a = grid
        if np.size(a) <= 10:
            self.logger.debug(
                np.array2string(np.asarray(a), precision=6, separator=", ")
            )
        else:
            self.logger.debug(f"size={np.size(a)}")
            self.logger.debug(f"min ={np.nanmin(a):.6g}")
            self.logger.debug(f"max ={np.nanmax(a):.6g}")
        self.logger.debug(f"head={np.asarray(a)[:3]}")
        self.logger.debug(f"tail={np.asarray(a)[-3:]}")
        return grid

    # ----------------------
    # 정적/동적 경계 (내장: 시작 로그 생략, 디버그만)
    # ----------------------
    def _bounds_base(self, o2: float | None) -> Tuple[float, float]:
        rc = self.rule_config
        lo_g, hi_g = self.opt_config.pump_bounds
        if o2 is None or pd.isna(o2):
            self.logger.debug(f"[base] O2 결측 → 전역 bounds 반환: ({lo_g}, {hi_g})")
            return (lo_g, hi_g)
        for up_to, (lo_opt, hi_opt) in rc.bounds_o2_table:
            if o2 <= up_to:
                lo = lo_g if lo_opt is None else float(lo_opt)
                hi = hi_g if hi_opt is None else float(hi_opt)
                self.logger.debug(f"[base] O2={o2} ≤ {up_to} → bounds=({lo}, {hi})")
                return (lo, hi)
        lo_opt, hi_opt = rc.bounds_o2_else
        if lo_opt is None and hi_opt is None:
            self.logger.debug(
                f"[base] O2={o2} else 케이스 → 고정 bounds=({lo_g}, {lo_g})"
            )
            return (lo_g, lo_g)
        lo = lo_g if lo_opt is None else float(lo_opt)
        hi = hi_g if hi_opt is None else float(hi_opt)
        self.logger.debug(f"[base] O2={o2} else 케이스 → bounds=({lo}, {hi})")
        return (lo, hi)

    def _bounds_dynamic(
        self,
        temp_in: float | None,
        temp_out: float | None,
        tgt: float | None,
        o2: float | None,
    ) -> Tuple[float, float]:
        rc = self.rule_config
        lo_g, hi_g = self.opt_config.pump_bounds

        temp_in = float(temp_in) if temp_in is not None else np.nan
        temp_out = float(temp_out) if temp_out is not None else np.nan
        tgt = float(tgt) if tgt is not None else np.nan
        o2 = float(o2) if o2 is not None else np.nan

        self.logger.debug(
            f"[dyn] 입력: temp_in={temp_in}, temp_out={temp_out}, tgt={tgt}, o2={o2}"
        )

        has_temp_in = pd.notna(temp_in)
        has_temp_out = pd.notna(temp_out)
        has_tgt = pd.notna(tgt)
        has_o2 = pd.notna(o2)

        if (has_temp_in and (temp_in <= rc.dyn_temp_in_low)) or (
            has_temp_out and (temp_out <= rc.dyn_temp_out_low)
        ):
            self.logger.debug(f"[dyn] 저온 조건 → ({lo_g},{lo_g})")
            return (lo_g, lo_g)
        if has_tgt and (tgt > rc.dyn_nox_high):
            self.logger.debug(f"[dyn] 고 NOx 조건 → ({hi_g},{hi_g})")
            return (hi_g, hi_g)
        if (has_tgt and (tgt <= rc.dyn_nox_low)) or (has_o2 and (o2 > rc.dyn_o2_high)):
            self.logger.debug(f"[dyn] 저 NOx 혹은 고 O2 조건 → ({lo_g},{lo_g})")
            return (lo_g, lo_g)
        if (
            (has_temp_in and (temp_in >= rc.dyn_temp_in_high))
            or (has_temp_out and (temp_out >= rc.dyn_temp_out_high))
            or (has_o2 and (o2 <= rc.dyn_o2_low))
        ):
            self.logger.debug(f"[dyn] 고온 혹은 저 O2 조건 → ({hi_g},{hi_g})")
            return (hi_g, hi_g)

        base = self._bounds_base(o2)
        self.logger.debug(f"[dyn] 동적 조건 미충족 → base bounds 사용: {base}")
        return base

    # ----------------------
    # 핵심: 단일 시점 추천 (모델 입력은 col_temp 사용)
    # ----------------------
    def predict_pump_hz(
        self,
        *,
        target_nox: float | None = None,
        pump_bounds: Tuple[float, float] | None = None,
        current_oxygen: float | None,
        current_temp: float | None,
        current_target: float | None,
        p_feasible: float | None = None,
        n_candidates: Optional[int] = None,
        round_to_int: bool | None = None,
    ) -> Dict[str, Any]:
        """GP+UCB 기반 단일 시점 펌프 Hz 추천 시작"""
        # 시작 로그 + 실행 파일
        self.logger.info(
            "펌프 주파수 추천 시작(GP+UCB): 단일 시점 입력에서 목표 NOx 만족 Hz 선택"
        )
        exec_file = os.path.basename(globals().get("__file__", "interactive"))
        self.logger.debug(f"실행 파일: {exec_file}")

        cc, oc = self.column_config, self.opt_config

        # 디버그용 전체 입력 로그
        self.logger.debug(f"입력값:")
        self.logger.debug(f"pump_bounds   ={pump_bounds}")
        self.logger.debug(f"target_nox    ={target_nox}")
        self.logger.debug(f"current_oxygen={current_oxygen}")
        self.logger.debug(f"current_temp  ={current_temp}")
        self.logger.debug(f"current_target={current_target}")
        self.logger.debug(f"p_feasible    ={p_feasible}")
        self.logger.debug(f"n_candidates  ={n_candidates}")
        self.logger.debug(f"round_to_int  ={round_to_int}")

        # 0) 구성값 정리
        tgt: float = oc.target_nox if target_nox is None else float(target_nox)
        lo, hi = (
            oc.pump_bounds
            if pump_bounds is None
            else (float(pump_bounds[0]), float(pump_bounds[1]))
        )
        p: float = oc.p_feasible if p_feasible is None else float(p_feasible)
        n_cand: int = oc.n_candidates if n_candidates is None else int(n_candidates)
        rint: bool = oc.round_to_int if round_to_int is None else bool(round_to_int)
        self.logger.info(f"파라미터:")
        self.logger.info(f"target={tgt}")
        self.logger.info(f"bounds=({lo},{hi})")
        self.logger.info(f"p={p}")
        self.logger.info(f"n_cand={n_cand}")
        self.logger.info(f"round_to_int={rint}")

        def _mk_grid() -> np.ndarray:
            return self._create_candidate_grid(lo, hi, n_cand, rint)

        def _payload(
            *,
            pump: float,
            grid: np.ndarray,
            o2: float | None,
            temp: float | None,
            mu: float | None = None,
            ucb: float | None = None,
            tgt_nox: float | None = None,
            safety_gap: float | None = None,
        ) -> Dict[str, Any]:
            return {
                cc.col_pred_mean: (np.nan if mu is None else float(mu)),
                cc.col_pred_ucb: (np.nan if ucb is None else float(ucb)),
                cc.col_target_nox: (
                    np.nan if tgt_nox is None or pd.isna(tgt_nox) else float(tgt_nox)
                ),
                cc.col_safety_gap: (
                    np.nan if safety_gap is None else float(safety_gap)
                ),
                cc.col_p_feasible: float(p),
                cc.col_round_flag: bool(rint),
                cc.col_n_candidates: (
                    None if n_candidates is None else int(n_candidates)
                ),
                cc.col_grid_min: float(grid.min()),
                cc.col_grid_max: float(grid.max()),
                cc.col_grid_size: int(grid.size),
                cc.col_o2: (np.nan if o2 is None or pd.isna(o2) else float(o2)),
                cc.col_temp: (np.nan if temp is None or pd.isna(temp) else float(temp)),
                cc.col_nox: (
                    np.nan
                    if current_target is None or pd.isna(current_target)
                    else float(current_target)
                ),
                cc.col_hz_out: float(pump),
            }

        # 1) 결측 → fallback
        if any(
            [
                current_oxygen is None or pd.isna(current_oxygen),
                current_temp is None or pd.isna(current_temp),
                current_target is None or pd.isna(current_target),
            ]
        ):
            self.logger.warning("입력 결측 감지 → fallback Hz 적용")
            fb = oc.fallback_hz if oc.fallback_hz is not None else oc.maximum_hz
            pump = float(np.clip(float(fb), lo, hi))
            grid = _mk_grid()
            self.logger.debug(f"fallback_hz={fb}")
            self.logger.debug(f"clip→pump={pump},")
            self.logger.debug(f"grid:")
            a = grid
            if np.size(a) <= 10:
                self.logger.debug(
                    np.array2string(np.asarray(a), precision=6, separator=", ")
                )
            else:
                self.logger.debug(f"size={np.size(a)}")
                self.logger.debug(f"min ={np.nanmin(a):.6g}")
                self.logger.debug(f"max ={np.nanmax(a):.6g}")
            self.logger.debug(f"head={np.asarray(a)[:3]}")
            self.logger.debug(f"tail={np.asarray(a)[-3:]}")
            self.logger.info("펌프 주파수 추천 완료")
            return _payload(
                pump=pump, grid=grid, o2=current_oxygen, temp=current_temp, tgt_nox=tgt
            )

        # 2) 정상 예측 경로
        z = oc.get_z_value(p)
        grid = _mk_grid()
        self.logger.debug(f"grid:")
        a = grid
        if np.size(a) <= 10:
            self.logger.debug(
                np.array2string(np.asarray(a), precision=6, separator=", ")
            )
        else:
            self.logger.debug(f"size={np.size(a)}")
            self.logger.debug(f"min ={np.nanmin(a):.6g}")
            self.logger.debug(f"max ={np.nanmax(a):.6g}")
        self.logger.debug(f"head={np.asarray(a)[:3]}")
        self.logger.debug(f"tail={np.asarray(a)[-3:]}")
        self.logger.debug(f"z={z}")

        Xcand = np.column_stack(
            [
                grid,
                np.full_like(grid, float(current_oxygen), dtype=float),
                np.full_like(grid, float(current_temp), dtype=float),
            ]
        )
        self.logger.debug(f"Xcand(shape={Xcand.shape}):")
        a = Xcand
        if np.size(a) <= 10:
            self.logger.debug(
                np.array2string(np.asarray(a), precision=6, separator=", ")
            )
        else:
            self.logger.debug(f"size={np.size(a)}")
            self.logger.debug(f"min ={np.nanmin(a):.6g}")
            self.logger.debug(f"max ={np.nanmax(a):.6g}")
        self.logger.debug(f"head={np.asarray(a)[:3]}")
        self.logger.debug(f"tail={np.asarray(a)[-3:]}")

        mu, sigma = self.model.predict(Xcand, return_std=True)
        ucb = mu + z * sigma
        self.logger.debug(f"mu:")
        a = mu
        if np.size(a) <= 10:
            self.logger.debug(
                np.array2string(np.asarray(a), precision=6, separator=", ")
            )
        else:
            self.logger.debug(f"size={np.size(a)}")
            self.logger.debug(f"min ={np.nanmin(a):.6g}")
            self.logger.debug(f"max ={np.nanmax(a):.6g}")
        self.logger.debug(f"head={np.asarray(a)[:3]}")
        self.logger.debug(f"tail={np.asarray(a)[-3:]}")
        self.logger.debug(f"sigma:")
        a = sigma
        if np.size(a) <= 10:
            self.logger.debug(
                np.array2string(np.asarray(a), precision=6, separator=", ")
            )
        else:
            self.logger.debug(f"size={np.size(a)}")
            self.logger.debug(f"min ={np.nanmin(a):.6g}")
            self.logger.debug(f"max ={np.nanmax(a):.6g}")
        self.logger.debug(f"head={np.asarray(a)[:3]}")
        self.logger.debug(f"tail={np.asarray(a)[-3:]}")
        self.logger.debug(f"ucb=mu+z*sigma:")
        a = ucb
        if np.size(a) <= 10:
            self.logger.debug(
                np.array2string(np.asarray(a), precision=6, separator=", ")
            )
        else:
            self.logger.debug(f"size={np.size(a)}")
            self.logger.debug(f"min ={np.nanmin(a):.6g}")
            self.logger.debug(f"max ={np.nanmax(a):.6g}")
        self.logger.debug(f"head={np.asarray(a)[:3]}")
        self.logger.debug(f"tail={np.asarray(a)[-3:]}")

        feasible = ucb <= tgt
        self.logger.debug(f"feasible(ucb<=target={tgt}):")
        a = feasible
        if np.size(a) <= 10:
            self.logger.debug(
                np.array2string(np.asarray(a), precision=6, separator=", ")
            )
        else:
            self.logger.debug(f"size={np.size(a)}")
            self.logger.debug(f"min ={np.nanmin(a):.6g}")
            self.logger.debug(f"max ={np.nanmax(a):.6g}")
        self.logger.debug(f"head={np.asarray(a)[:3]}")
        self.logger.debug(f"tail={np.asarray(a)[-3:]}")

        idx = int(np.where(feasible)[0][0]) if np.any(feasible) else int(np.argmin(ucb))
        pump = float(grid[idx])
        mu_i = float(mu[idx])
        ucb_i = float(ucb[idx])
        gap = float(tgt - ucb_i)

        self.logger.info(
            f"선택: idx={idx}, pump={pump}, μ={mu_i:.6f}, UCB={ucb_i:.6f}, gap={gap:.6f}, feasible_any={bool(np.any(feasible))}"
        )
        self.logger.info("펌프 주파수 추천 완료")

        return _payload(
            pump=pump,
            grid=grid,
            o2=current_oxygen,
            temp=current_temp,
            mu=mu_i,
            ucb=ucb_i,
            tgt_nox=tgt,
            safety_gap=gap,
        )

    # ----------------------
    # 후처리: 규칙 적용
    # ----------------------
    def add_rule_columns(self, df_recommend: pd.DataFrame) -> pd.DataFrame:
        """규칙 기반 후처리 시작: O2/동적 경계 적용"""
        self.logger.info("규칙 기반 후처리 시작: O2/동적 경계 적용")
        exec_file = os.path.basename(globals().get("__file__", "interactive"))
        self.logger.debug(f"실행 파일: {exec_file}")

        cc = self.column_config
        lo_g, hi_g = self.opt_config.pump_bounds
        out = df_recommend.copy()

        base_col = (
            cc.col_hz_raw_out if cc.col_hz_raw_out in out.columns else cc.col_hz_out
        )
        base = out[base_col].astype(float).to_numpy()
        o2 = out[cc.col_o2].astype(float).to_numpy()

        temp_in_src = out.get(
            cc.col_inner_temp, pd.Series(index=out.index, dtype=float)
        )
        if temp_in_src is None:
            temp_in_src = pd.Series(index=out.index, dtype=float)
        temp_in_series = temp_in_src.astype(float).where(
            ~temp_in_src.isna(), out.get(cc.col_temp, np.nan)
        )

        temp_in = temp_in_series.astype(float).to_numpy()
        temp_out = (
            out.get(cc.col_outer_temp, pd.Series(np.nan, index=out.index))
            .astype(float)
            .to_numpy()
        )
        nox = out[cc.col_nox].astype(float).to_numpy()

        self.logger.debug(f"입력 변수 값:")
        self.logger.debug(f"`hz_raw`  : %.6g", float(np.ravel(base)[0]))
        self.logger.debug(f"`o2`      : %.6g", float(np.ravel(o2)[0]))
        self.logger.debug(f"`temp_in` : %.6g", float(np.ravel(temp_in)[0]))
        self.logger.debug(f"`temp_out`: %.6g", float(np.ravel(temp_out)[0]))
        self.logger.debug(f"`nox`     : %.6g", float(np.ravel(nox)[0]))

        lo_o2 = np.empty_like(base)
        hi_o2 = np.empty_like(base)
        lo_dyn = np.empty_like(base)
        hi_dyn = np.empty_like(base)

        for i in range(base.size):
            lo1, hi1 = self._bounds_base(o2[i])
            lo2, hi2 = self._bounds_dynamic(temp_in[i], temp_out[i], nox[i], o2[i])
            lo_o2[i], hi_o2[i] = lo1, hi1
            lo_dyn[i], hi_dyn[i] = lo2, hi2
            self.logger.debug(
                f"[i={i}] base_bounds=({lo1},{hi1}), dyn_bounds=({lo2},{hi2}), "
                f"base={base[i]}, o2={o2[i]}, temp_in={temp_in[i]}, temp_out={temp_out[i]}, nox={nox[i]}"
            )

        tightened_o2 = (lo_o2 > lo_g) | (hi_o2 < hi_g)
        tightened_dyn = (lo_dyn > lo_g) | (hi_dyn < hi_g)

        hz_o2 = np.where(tightened_o2, np.clip(base, lo_o2, hi_o2), base)
        hz_all = np.where(tightened_dyn, np.clip(base, lo_dyn, hi_dyn), base)

        out[cc.col_hz_init_rule] = hz_o2.astype(float)
        out[cc.col_hz_full_rule] = hz_all.astype(float)

        self.logger.info(
            f"후처리 적용 요약: tightened_o2={int(tightened_o2.sum())}/{base.size}, "
            f"tightened_dyn={int(tightened_dyn.sum())}/{base.size}"
        )
        if base.size == 1:
            self.logger.debug(f"최종 행 상세: base={base[0]}")
            self.logger.debug(f"O2_bounds=({lo_o2[0]},{hi_o2[0]})")
            self.logger.debug(f"hz_init_rule={hz_o2[0]}")
            self.logger.debug(f"DYN_bounds=({lo_dyn[0]},{hi_dyn[0]})")
            self.logger.debug(f"hz_full_rule={hz_all[0]}")

        self.logger.info("규칙 기반 후처리 완료")
        return out

    # ----------------------
    # 파이프라인 (단일 시점)
    # ----------------------
    def run_pipeline(
        self, df: pd.DataFrame, *, at_time: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """실시간 파이프라인 시작: 최신(또는 지정) 시점에서 추천/후처리 실행"""
        self.logger.info("실시간 파이프라인 실행 시작: 단일 시점 추천")
        exec_file = os.path.basename(globals().get("__file__", "interactive"))
        self.logger.debug(f"실행 파일: {exec_file}")

        cc = self.column_config
        rc = self.rule_config
        oc = self.opt_config

        self.logger.debug(f"입력 df 크기: {df.shape}")
        if df.empty:
            msg = "입력 df가 비어 있습니다."
            self.logger.error(msg)
            raise ValueError(msg)

        d = df.copy()
        d[cc.col_datetime] = pd.to_datetime(d[cc.col_datetime], errors="coerce")
        before = len(d)
        d = d.dropna(subset=[cc.col_datetime])
        self.logger.debug(f"datetime 파싱/필터: {before} → {len(d)}행")
        if d.empty:
            msg = f"{cc.col_datetime} 파싱 후 유효한 행이 없습니다."
            self.logger.error(msg)
            raise ValueError(msg)

        # 기준 시점 선택
        if at_time is not None:
            at_time = pd.to_datetime(at_time)
            sel = d[d[cc.col_datetime] == at_time]
            if sel.empty:
                msg = f"at_time={at_time} 에 해당하는 행이 없습니다."
                self.logger.error(msg)
                raise ValueError(msg)
            row = sel.sort_values(cc.col_datetime).iloc[0]
            self.logger.info(f"선택 시점: {at_time}")
        else:
            last_t = d[cc.col_datetime].max()
            row = d[d[cc.col_datetime] == last_t].sort_values(cc.col_datetime).iloc[0]
            self.logger.info(f"선택 시점(최신): {last_t}")

        # 결측이면 fallback 경고
        req_cols = [cc.col_o2, cc.col_temp, cc.col_nox]
        missing = [c for c in req_cols if pd.isna(row.get(c, np.nan))]
        if missing:
            self.logger.warning(f"입력 결측으로 fallback 적용: {missing}")

        # 선택된 행 상세(디버깅)
        self.logger.debug(f"선택 행 상세:")
        self.logger.debug(f"datetime={row.get(cc.col_datetime)}")
        self.logger.debug(f"O2={row.get(cc.col_o2)}")
        self.logger.debug(f"Temp={row.get(cc.col_temp)}")
        self.logger.debug(f"NOx={row.get(cc.col_nox)}")
        self.logger.debug(f"InnerTemp={row.get(cc.col_inner_temp, np.nan)}")
        self.logger.debug(f"OuterTemp={row.get(cc.col_outer_temp, np.nan)}")

        # 단일 시점 추천
        res = self.predict_pump_hz(
            target_nox=oc.target_nox,
            pump_bounds=oc.pump_bounds,
            current_oxygen=float(row[cc.col_o2]) if pd.notna(row[cc.col_o2]) else None,
            current_temp=(
                float(row[cc.col_temp])
                if pd.notna(row.get(cc.col_temp, np.nan))
                else None
            ),
            current_target=(
                float(row[cc.col_nox]) if pd.notna(row[cc.col_nox]) else None
            ),
            p_feasible=oc.p_feasible,
            n_candidates=oc.n_candidates,
            round_to_int=oc.round_to_int,
        )
        self.logger.debug(f"추천 결과 payload: {res}")

        # 결과 한 줄 구성
        rec = {cc.col_datetime: row[cc.col_datetime]}
        rec.update(res)
        if cc.col_inner_temp in d.columns:
            rec[cc.col_inner_temp] = row.get(cc.col_inner_temp, np.nan)
        if cc.col_outer_temp in d.columns:
            rec[cc.col_outer_temp] = row.get(cc.col_outer_temp, np.nan)

        df_rec = pd.DataFrame([rec])

        # 모델 결과 컬럼명 통일: raw → raw_out
        if cc.col_hz_out in df_rec.columns and cc.col_hz_raw_out not in df_rec.columns:
            df_rec = df_rec.rename(columns={cc.col_hz_out: cc.col_hz_raw_out})

        # 규칙 후처리
        df_final = self.add_rule_columns(df_rec)

        # col_datetime 선두 보장 및 한 줄 반환
        cols = list(df_final.columns)
        if cols and cols[0] != cc.col_datetime:
            cols = [cc.col_datetime] + [c for c in cols if c != cc.col_datetime]
            df_final = df_final[cols]

        self.logger.info("요소수 펌프 Hz 최적화 완료")
        self.logger.debug("최종 결과 1행:")
        text = pformat(df_final.iloc[0].to_dict(), width=120, compact=False)
        for line in text.splitlines():
            self.logger.info(line)
        return df_final.iloc[[0]]
