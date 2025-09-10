# ======================
# data_loader.py
# ======================
"""
Data loader for SKEP Urea Control System
(기존 메서드만 유지, 한국어 로그 + 단계별 시작/종료 로그 추가)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from time import perf_counter as pc
import os
import logging
import pandas as pd
from pprint import pformat

from config.column_config import ColumnConfig
from config.preprocessing_config import (
    GPTrainPreprocessingConfig,
    InferPreprocessingConfig,
)
from utils.logger import get_logger, LoggerConfig


@dataclass
class DataLoader:
    """데이터 로더 (학습/추론 공용) - 기존 메서드만 유지, 내부 로그 보강(한국어)"""

    column_config: Optional[ColumnConfig] = None
    preprocessing_config: Optional[GPTrainPreprocessingConfig] = None

    # 로거 인스턴스(설정은 preprocessing_config.logger_cfg에서 주입)
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.cc = self.column_config or ColumnConfig()
        self.preprocessing_config = (
            self.preprocessing_config or GPTrainPreprocessingConfig()
        )
        logger_cfg = getattr(
            self.preprocessing_config,
            "logger_cfg",
            LoggerConfig(name="DataLoader", level=logging.INFO),
        )
        self.logger = get_logger(logger_cfg)

        self.logger.debug("[INIT] DataLoader 생성 완료")
        self.logger.debug("로거=%s", logger_cfg.name)
        self.logger.debug("레벨=%s", logging.getLevelName(logger_cfg.level))
        self.logger.debug("시간컬럼=%s", self.cc.col_datetime)
        self.logger.debug("피처=%s", self.cc.gp_feature_columns)
        self.logger.debug("타깃=%s", self.cc.target_column)

    # -----------------------------
    # Public APIs
    # -----------------------------
    def load_data_by_time(
        self,
        filepath: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> pd.DataFrame:
        """start, end time 기준 데이터 로드"""
        t0 = pc()
        self.logger.info("[START] 데이터 로드 시작")
        self.logger.info("path=%s", filepath)
        cols = self._training_columns()
        self.logger.debug("요구 컬럼=%s", cols)
        self.logger.debug("시간필터 start=%s end=%s", start_time, end_time)

        df = self._load_any(filepath, columns=cols)
        self.logger.info("원본 크기=%s", df.shape)

        self.logger.debug("시간 파싱 시작")
        df = self._ensure_datetime(df, self.cc.col_datetime)
        self.logger.debug("시간 파싱 종료")

        if start_time or end_time:
            self.logger.debug("시간 필터링 시작")
        df = self._filter_by_time_range(df, self.cc.col_datetime, start_time, end_time)
        if start_time or end_time:
            self.logger.debug("시간 필터링 종료")

        self.logger.info("[END] 데이터 로드 종료")
        self.logger.info("총 소요=%.3fs", pc() - t0)
        self.logger.info("최종 크기=%s", df.shape)
        self.logger.info(
            "범위=[%s ~ %s]",
            (
                str(df[self.cc.col_datetime].min())
                if self.cc.col_datetime in df.columns and len(df)
                else None
            ),
            (
                str(df[self.cc.col_datetime].max())
                if self.cc.col_datetime in df.columns and len(df)
                else None
            ),
        )
        return df.reset_index(drop=True)

    def load_data_by_timedelta(
        self,
        filepath: str,
        *,
        run_time: Optional[str] = None,
        window_sec: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        run_time과 window_sec 기준 데이터 로드(실행시점 기준 window_sec 전 ~ 실행시점)

        Parameters
        ----------
        filepath : str
            데이터 파일 경로
        run_time : str | None
            실행 기준시각 (없으면 전체에서 max(datetime)를 기준으로 함)
        window_sec : int | None
            윈도우 크기(초). None이면 InferPreprocessingConfig.ffill_limit_sec 또는 기본 20초 사용.

        Returns
        -------
        pd.DataFrame
            [run_time - window_sec, run_time] 구간의 데이터프레임(시간 정렬)
        """
        t0 = pc()
        self.logger.info("[START] 데이터 로드 시작")
        self.logger.info("path=%s", filepath)
        self.logger.info("run_time=%s", run_time)
        self.logger.info("window_sec=%s", window_sec)

        cols = self._inference_columns()
        self.logger.debug("요구 컬럼=%s", cols)

        df = self._load_any(filepath, columns=cols)
        self.logger.info("원본 크기=%s", df.shape)

        self.logger.debug("시간 파싱 시작")
        df = self._ensure_datetime(df, self.cc.col_datetime)
        self.logger.debug("시간 파싱 종료")

        if self.cc.col_datetime not in df.columns:
            msg = (
                f"'{self.cc.col_datetime}' 컬럼이 없어 추론 시점을 선택할 수 없습니다."
            )
            self.logger.error("%s", msg)
            raise ValueError(msg)

        ts = pd.to_datetime(df[self.cc.col_datetime], errors="coerce")
        mask = ~ts.isna()
        df = df.loc[mask].copy()
        df[self.cc.col_datetime] = ts[mask].values

        if run_time is not None:
            rt = pd.to_datetime(run_time, errors="raise")
        else:
            rt = df[self.cc.col_datetime].max()
            self.logger.info("run_time 미지정 → 데이터 max 시각 사용: %s", rt)

        start_ts = rt - pd.Timedelta(seconds=int(window_sec))
        before = len(df)
        df = df.loc[
            (df[self.cc.col_datetime] >= start_ts) & (df[self.cc.col_datetime] <= rt)
        ]
        self.logger.info("윈도우 필터 [%s, %s]: %d → %d", start_ts, rt, before, len(df))

        if df.empty:
            msg = f"윈도우 구간에 데이터가 없습니다: [{start_ts}, {rt}]"
            self.logger.error("%s", msg)
            raise ValueError(msg)

        df = df.sort_values(self.cc.col_datetime).reset_index(drop=True)
        self.logger.info("[END] 데이터 로드 종료")
        self.logger.info("총 소요=%.3fs", pc() - t0)
        self.logger.info("최종 크기=%s", df.shape)
        self.logger.info(
            "범위=[%s ~ %s]",
            (
                str(df[self.cc.col_datetime].min())
                if self.cc.col_datetime in df.columns and len(df)
                else None
            ),
            (
                str(df[self.cc.col_datetime].max())
                if self.cc.col_datetime in df.columns and len(df)
                else None
            ),
        )
        return df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        t0 = pc()
        self.logger.info("[START] 데이터 유효성 점검 시작")
        cc = self.cc
        train_req = self._training_required()
        infer_req = self._inference_required()

        def _missing(cols: List[str]) -> List[str]:
            return [c for c in cols if c not in df.columns]

        def _na_ratio(cols: List[str]) -> Dict[str, float]:
            out = {}
            for c in cols:
                out[c] = float(df[c].isna().mean()) if c in df.columns else 1.0
            return out

        info = {
            "shape": tuple(df.shape),
            "has_datetime": cc.col_datetime in df.columns,
            "missing_for_training": _missing(train_req),
            "missing_for_inference": _missing(infer_req),
            "na_ratio_training": _na_ratio([c for c in train_req if c in df.columns]),
            "na_ratio_inference": _na_ratio([c for c in infer_req if c in df.columns]),
        }

        if cc.col_datetime in df.columns and len(df):
            ts = pd.to_datetime(df[cc.col_datetime], errors="coerce")
            info["time_coverage"] = {
                "min": str(ts.min()),
                "max": str(ts.max()),
                "n_na_ts": int(ts.isna().sum()),
            }
        else:
            info["time_coverage"] = None

        text = pformat(info, width=120, compact=False)
        for line in text.splitlines():
            self.logger.info(line)
        self.logger.info("[END] 데이터 유효성 점검 종료 | 총 소요=%.3fs", pc() - t0)
        return info

    # -----------------------------
    # Internal helpers (원래 있던 것만 유지)
    # -----------------------------
    def _training_required(self) -> List[str]:
        return list(
            dict.fromkeys(
                self.cc.gp_feature_columns
                + self.cc.lgbm_feature_columns
                + [self.cc.target_column]
            )
        )

    def _training_columns(self) -> List[str]:
        cols = [self.cc.col_datetime] + self._training_required()
        cols += [self.cc.col_ai, self.cc.col_act_status, self.cc.col_inc_status]
        cols += (
            self.cc.cols_temp
            + self.cc.cols_icf_tms
            + self.cc.cols_tms_value
            + self.cc.cols_tms_eq_status
        )
        return list(dict.fromkeys(cols))

    def _inference_required(self) -> List[str]:
        return list(dict.fromkeys(self.cc.gp_feature_columns + [self.cc.target_column]))

    def _inference_columns(self) -> List[str]:
        cols = [self.cc.col_datetime] + self._inference_required()
        cols += self.cc.cols_temp
        return list(dict.fromkeys(cols))

    def _load_any(self, path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        t0 = pc()
        ext = os.path.splitext(path)[1].lower()
        self.logger.debug(
            "[LOAD] 시작 | path=%s | 확장자=%s | 요청컬럼=%s", path, ext, columns
        )

        def _slice(df: pd.DataFrame) -> pd.DataFrame:
            if columns:
                cols_exist = [c for c in columns if c in df.columns]
                missing = [c for c in columns if c not in df.columns]
                if missing:
                    self.logger.warning("[LOAD] 요청 컬럼 일부 누락: %s", missing)
                return df[cols_exist]
            return df

        try:
            if ext in (".parquet", ".pq"):
                try:
                    df = pd.read_parquet(path, columns=columns)
                except Exception as e:
                    self.logger.debug(
                        "[LOAD] parquet 선택 로드 실패 → 전체 로드 후 슬라이스: %r", e
                    )
                    df = _slice(pd.read_parquet(path))
            elif ext in (".csv", ".txt"):
                try:
                    df = pd.read_csv(path, usecols=columns)
                except Exception as e:
                    self.logger.debug(
                        "[LOAD] csv 선택 로드 실패 → 전체 로드 후 슬라이스: %r", e
                    )
                    df = _slice(pd.read_csv(path))
            elif ext in (".feather", ".ft"):
                df = _slice(pd.read_feather(path))
            else:
                raise ValueError(f"지원하지 않는 확장자: {ext}")
        except Exception:
            self.logger.exception("[LOAD] 파일 로드 실패: %s", path)
            raise

        self.logger.debug("[LOAD] 종료 | 크기=%s | 소요=%.3fs", df.shape, pc() - t0)
        return df

    def _ensure_datetime(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        self.logger.debug("[TIME] 시작 | 컬럼=%s", time_col)
        if time_col in df.columns:
            before_na = int(pd.isna(df[time_col]).sum())
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            after_na = int(pd.isna(df[time_col]).sum())
            self.logger.debug(
                "[TIME] '%s' 파싱 NA: %d → %d", time_col, before_na, after_na
            )
        else:
            self.logger.info("[TIME] '%s' 컬럼 없음 → 시간 파싱 생략", time_col)
        self.logger.debug("[TIME] 종료")
        return df

    def _filter_by_time_range(
        self,
        df: pd.DataFrame,
        time_col: str,
        start_time: Optional[str],
        end_time: Optional[str],
    ) -> pd.DataFrame:
        self.logger.debug("[TIME/FILTER] 시작 | start=%s end=%s", start_time, end_time)
        if time_col not in df.columns:
            self.logger.info("[TIME/FILTER] '%s' 컬럼 없음 → 시간 필터 생략", time_col)
            self.logger.debug("[TIME/FILTER] 종료")
            return df

        d = df.copy()
        if start_time:
            before = len(d)
            d = d[d[time_col] >= pd.to_datetime(start_time)]
            self.logger.debug(
                "[TIME/FILTER] start_time>=%s : %d → %d", start_time, before, len(d)
            )
        if end_time:
            before = len(d)
            d = d[d[time_col] <= pd.to_datetime(end_time)]
            self.logger.debug(
                "[TIME/FILTER] end_time<=%s : %d → %d", end_time, before, len(d)
            )
        d = d.sort_values(time_col).reset_index(drop=True)
        self.logger.debug("[TIME/FILTER] 종료")
        return d
