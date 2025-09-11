#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SKEP Urea Control System - Pump Optimization Script
실행시점(run_time) 기준 최신 1행으로 최적화
- 실행 시각 입력 지원:
  1) --run_time "YYYY-MM-DD HH:MM:SS"
  2) --run_date "YYYY-MM-DD" + --run_time "HH:MM:SS"
"""

import os
import sys
import re
import argparse
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from time import time

# --- runtime path injection (root/src) ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# repo 구조 임포트
from src.models.gaussian_process import GaussianProcessNOxModel
from src.models.lgbm import LGBMNOxModel
from src.optimization.pump_optimizer import PumpOptimizer
from src.optimization.pump_hz_adjuster import LGBMPumpHzAdjuster
from src.data_processing.data_loader import DataLoader
from src.data_processing.preprocessor import Preprocessor, LGBMFeaturePreprocessor
from config.column_config import ColumnConfig
from config.preprocessing_config import (
    InferPreprocessingConfig,
    LGBMInferPreprocessingConfig,
)
from config.model_config import GPModelConfig, LGBMModelConfig
from config.optimization_config import OptimizationConfig
from config.rule_config import RuleConfig
from utils.logger import LoggerConfig  # 추가


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SKEP Urea Pump Optimization (latest single row)"
    )
    parser.add_argument(
        "--plant_code", type=str, default=None, help="플랜트 코드 (선택)"
    )
    parser.add_argument("--gp_model_path", type=str, required=True)
    parser.add_argument("--lgbm_model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    # 최종 Hz 추천 값으로 뭘 쓸지 (col_hz_final; step 0)
    # - Hz 추천 1,2,3,4 중 택 1 -> 웨이블 등에 표시
    parser.add_argument("--hz_step_num", type=int, required=True)

    # 실행시각 입력 (분리 입력 지원 / run_clock 제거)
    parser.add_argument(
        "--run_time",
        type=str,
        default=None,
        help='1) "YYYY-MM-DD HH:MM:SS" 또는 2) --run_date + --run_time("HH:MM:SS")',
    )
    parser.add_argument(
        "--run_date",
        type=str,
        default=None,
        help='옵션: "YYYY-MM-DD" (분리 입력시 날짜)',
    )
    return parser.parse_args()


def build_column_config(plant_code: str | None) -> ColumnConfig:
    """plant_code가 있으면 적용, 없으면 기본 config"""
    return ColumnConfig(plant_code=plant_code) if plant_code else ColumnConfig()


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("\u00a0", " ")).strip()


def parse_run_timestamp(
    rt_str: str | None, date_str: str | None
) -> tuple[pd.Timestamp, str]:
    """
    실행시각 파서 (run_clock 제거 버전).
    우선순위:
      1) run_date 있고 run_time이 HH:MM:SS이면 -> 결합
      2) run_time이 날짜 포함(YYYY-MM-DD ...)이면 -> 그대로
      3) run_date만 있으면 -> EOD(23:59:59.999999)
      4) run_time이 HH:MM:SS '단독'이면 -> 오류(혼동 방지)
      5) 기타 -> pandas 파싱
    """
    s_date = _normalize_spaces(date_str) if date_str else None
    s_rt = _normalize_spaces(rt_str) if rt_str else None

    # 1) 분리 입력: 날짜 + 시간 결합
    if s_date and s_rt and re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", s_rt):
        cand = f"{s_date} {s_rt}"
        src = "run_date+run_time"
    # 2) run_time이 날짜 포함
    elif s_rt and re.search(r"\d{4}-\d{2}-\d{2}", s_rt):
        cand = s_rt
        src = "run_time"
    # 3) run_date만 제공(EOD 보정)
    elif s_date and not s_rt:
        d_only = datetime.strptime(s_date, "%Y-%m-%d")
        dt = d_only.replace(hour=23, minute=59, second=59, microsecond=999999)
        return pd.Timestamp(dt), "run_date(EOD)"
    # 4) 시간만 단독으로 들어옴 -> 명시적 오류
    elif s_rt and re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", s_rt) and not s_date:
        raise ValueError(
            '시간만(--run_time "HH:MM:SS") 입력 시에는 --run_date "YYYY-MM-DD"를 함께 제공해야 합니다.'
        )
    else:
        if not s_rt:
            raise ValueError(
                "run_time 또는 run_date(+run_time) 중 하나는 반드시 제공되어야 합니다."
            )
        cand = s_rt
        src = "run_time"

    # 포맷 고정 파싱
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y.%m.%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(cand, fmt)
            return pd.Timestamp(dt), src
        except ValueError:
            pass

    # 날짜만이면 EOD
    try:
        d_only = datetime.strptime(cand, "%Y-%m-%d")
        dt = d_only.replace(hour=23, minute=59, second=59, microsecond=999999)
        return pd.Timestamp(dt), src
    except ValueError:
        # 최후수단
        return pd.to_datetime(cand, errors="raise"), src


def main():
    args = parse_args()
    rt, rt_src = parse_run_timestamp(args.run_time, args.run_date)

    # ColumnConfig & Model/Optimization Config
    cc = build_column_config(args.plant_code)
    mc = GPModelConfig()
    oc = OptimizationConfig()

    print(f"[INFO] Loading model: {args.gp_model_path}")
    print(f"[INFO] Loading data : {args.data_path}")
    print(f"[INFO] Run time     : {rt} (parsed from {rt_src})")
    print(f"[INFO] plant_code preset  : {args.plant_code} (col_temp={cc.col_temp})")

    # 모델 로드
    gp = GaussianProcessNOxModel(column_config=cc, model_config=mc).load(
        args.gp_model_path
    )
    print(gp.model.kernel_)  # 최적화된 커널 전체
    print(gp.model.kernel_.theta)  # log-space 파라미터 값
    print(gp.model.kernel_.get_params())  # 딕셔너리 형태 파라미터

    # -----------------------------------------------------------
    # 실행시점 기준 "20초 전 ~ 실행시점" 구간 로드 + 제한 ffill + 마지막 1행 선택
    #   - 20초 및 resample_sec은 InferPreprocessingConfig에서 가져옴
    # -----------------------------------------------------------
    prep_infer_cfg = InferPreprocessingConfig(
        column_config=cc
    )  # ffill_limit_sec/resample_sec 사용
    dl = DataLoader(cc)  # 로거/세부는 내부 기본 사용
    pp = Preprocessor()

    # 1) 윈도우 로드 (run_time - ffill_limit_sec ~ run_time)
    df_win = dl.load_data_by_timedelta(
        args.data_path,
        run_time=str(rt),
        window_sec=600,  # prep_infer_cfg.ffill_limit_sec
    )

    # 2) 제한 ffill 후 실행시점(이하) 마지막 1행만 선택
    df_one = pp.make_infer_ffill(
        df_win,
        require_full_index=True,  # full index 생성 후 제한 ffill 권장
        # 로깅
        logger_cfg=LoggerConfig(
            name="Preprocessing", level=logging.INFO, refresh_handlers=False
        ),
    )

    # 최적화
    opt = PumpOptimizer(model=gp, column_config=cc, opt_config=oc)
    row = df_one.iloc[-1]
    result = opt.predict_pump_hz(
        target_nox=oc.target_nox,
        pump_bounds=oc.pump_bounds,
        current_oxygen=float(row[cc.col_o2]) if pd.notna(row[cc.col_o2]) else None,
        current_temp=float(row[cc.col_temp]) if pd.notna(row[cc.col_temp]) else None,
        current_target=float(row[cc.col_nox]) if pd.notna(row[cc.col_nox]) else None,
        p_feasible=oc.p_feasible,
        n_candidates=oc.n_candidates,
        round_to_int=oc.round_to_int,
    )
    result[cc.col_datetime] = row[cc.col_datetime]
    if cc.col_inner_temp in df_one.columns:
        result[cc.col_inner_temp] = row.get(cc.col_inner_temp, pd.NA)
    if cc.col_outer_temp in df_one.columns:
        result[cc.col_outer_temp] = row.get(cc.col_outer_temp, pd.NA)

    df_rec = pd.DataFrame([result])
    if cc.col_hz_out in df_rec.columns and cc.col_hz_raw_out not in df_rec.columns:
        df_rec = df_rec.rename(columns={cc.col_hz_out: cc.col_hz_raw_out})

    df_with_rules = opt.add_rule_columns(df_rec)

    # 저장
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    ext = os.path.splitext(args.output_path)[1].lower()
    if ext in (".parquet", ".pq"):
        df_with_rules.to_parquet(args.output_path, index=False)
    elif ext in (".csv", ".txt"):
        df_with_rules.to_csv(args.output_path, index=False)
    else:
        raise ValueError(f"Unsupported output extension: {ext}")

    print(f"[INFO] Saved results to: {args.output_path}")
    print("\n=== Optimization Summary (single row) ===")
    if cc.col_hz_raw_out in df_with_rules:
        print(f"  Raw Hz     : {float(df_with_rules.iloc[0][cc.col_hz_raw_out]):.2f}")
    print(f"  O2 rule Hz : {float(df_with_rules.iloc[0][cc.col_hz_init_rule]):.2f}")
    print(f"  Full rule  : {float(df_with_rules.iloc[0][cc.col_hz_full_rule]):.2f}")
    print("[INFO] Optimization completed.")

    # # -----------------------------------------------------------
    # # LGBM
    # # -----------------------------------------------------------

    # # config
    # lgbm_prep_infer_cfg = LGBMInferPreprocessingConfig(column_config=cc)
    # lgbm_cols_x_original = cc.lgbm_feature_columns

    # # Infer용 전처리: 요약통계량 Feature 생성
    # lgbm_pp = LGBMFeaturePreprocessor(lgbm_prep_infer_cfg)  # infer cfg를 써도 동작함
    # lgbm_suggested_df, lgbm_cols_x_stat = lgbm_pp.make_interval_features(df_one)
    # lgbm_suggested_df = lgbm_suggested_df.iloc[-1:,]
    # lgbm_suggested_df[cc.col_hz_raw_out] = float(
    #     df_with_rules.iloc[-1][cc.col_hz_raw_out]
    # )
    # lgbm_suggested_df[cc.col_hz_init_rule] = float(
    #     df_with_rules.iloc[-1][cc.col_hz_init_rule]
    # )
    # lgbm_suggested_df[cc.col_hz_full_rule] = float(
    #     df_with_rules.iloc[-1][cc.col_hz_full_rule]
    # )

    # # === NOx 예측값 & Hz 보정값 생성 ===

    # # 원본 + 요약통계 피처를 합쳐 "최종 학습 피처"로 사용
    # lgbm_mc = LGBMModelConfig(
    #     # ✅ 입력 컬럼(생성 시 주입)
    #     lgbm_feature_columns_original=list(lgbm_cols_x_original),
    #     lgbm_feature_columns_summary=list(lgbm_cols_x_stat),
    #     # 저장 경로(Booster 포함)
    #     # lgbm_model_path=args.lgbm_model_path,
    #     # meta_path=args.lgbm_model_path,
    #     native_model_path=args.lgbm_model_path,
    # )
    # lgbm_adjuster = LGBMPumpHzAdjuster(
    #     column_config=cc,
    #     model_config=lgbm_mc,
    #     rule_config=RuleConfig(),
    #     optimization_config=oc,
    # )
    # lgbm_suggested_df = lgbm_adjuster.predict_and_adjust(
    #     lgbm_suggested_df, return_flags=True
    # )

    # print(50 * "=")
    # print("\n=== LGBM Summary (single row) ===")
    # print(
    #     f"Predicted NOx Value: {float(lgbm_suggested_df.iloc[0][cc.col_lgbm_db_pred_nox]):.2f}"
    # )
    # print(
    #     f"Adjusted Pump Hz: {float(lgbm_suggested_df.iloc[0][cc.col_lgbm_db_hz_lgbm_adj]):.2f}"
    # )
    # print("[INFO] End.")
    # print(50 * "=")

    # final_output_df = lgbm_suggested_df[
    #     [
    #         cc.col_datetime,
    #         cc.col_hz_raw_out,
    #         cc.col_hz_init_rule,
    #         cc.col_hz_full_rule,
    #         cc.col_lgbm_db_hz_lgbm_adj,
    #         cc.col_lgbm_db_pred_nox,
    #     ]
    # ].copy()
    # if int(args.hz_step_num) == 1:
    #     final_output_df[cc.col_hz_final] = float(
    #         lgbm_suggested_df.iloc[-1][cc.col_hz_raw_out]
    #     )
    # elif int(args.hz_step_num) == 2:
    #     final_output_df[cc.col_hz_final] = float(
    #         lgbm_suggested_df.iloc[-1][cc.col_hz_init_rule]
    #     )
    # elif int(args.hz_step_num) == 3:
    #     final_output_df[cc.col_hz_final] = float(
    #         lgbm_suggested_df.iloc[-1][cc.col_hz_full_rule]
    #     )
    # elif int(args.hz_step_num) == 4:
    #     final_output_df[cc.col_hz_final] = float(
    #         lgbm_suggested_df.iloc[-1][cc.col_lgbm_db_hz_lgbm_adj]
    #     )
    # else:
    #     raise ValueError(f"Hz 추천은 1,2,3,4 중 하나를 선택해야 합니다.")
    # print(final_output_df.T)


if __name__ == "__main__":
    main()
