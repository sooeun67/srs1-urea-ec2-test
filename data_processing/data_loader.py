"""
Data loader for SKEP Urea Control System
"""
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any
import pandas as pd

from config.column_config import ColumnConfig


class DataLoader:
    """데이터 로더 (학습/추론 공용)"""

    def __init__(self, column_config: Optional[ColumnConfig] = None):
        self.cc = column_config or ColumnConfig()

    # -----------------------------
    # Public APIs
    # -----------------------------
    def load_training_data(
        self,
        filepath: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        모델 학습용 데이터 로드
        최소 컬럼 = feature_columns + [target_column] + [col_datetime] (+ inner/outer optional)
        """
        cols = self._training_columns()
        df = self._load_any(filepath, columns=cols)
        df = self._ensure_datetime(df, self.cc.col_datetime)
        df = self._filter_by_time_range(df, self.cc.col_datetime, start_time, end_time)

        # col_ai 값이 1인 데이터만 필터링
        if hasattr(self.cc, "col_ai") and self.cc.col_ai in df.columns:
            df = df.loc[df[self.cc.col_ai] == 1].copy()
            
        return df.reset_index(drop=True)

    def load_inference_data(
        self,
        filepath: str,
        *,
        run_time: Optional[str] = None,
        require_hz: bool = False,
    ) -> pd.DataFrame:
        """
        모델 추론용 데이터 로드 (가장 최근 1행만 반환)
        - 최소 컬럼 = feature_columns + [target_column] + [col_datetime] (+ inner/outer optional)
        - run_time이 주어지면: 해당 시각 이하(<= run_time) 데이터 중 가장 최근 1행
        - run_time이 없으면: 전체 중 가장 최근 1행
        """
        cols = self._inference_columns()
        df = self._load_any(filepath, columns=cols)
        df = self._ensure_datetime(df, self.cc.col_datetime)

        if self.cc.col_datetime not in df.columns:
            raise ValueError(f"'{self.cc.col_datetime}' column is required for inference timestamp selection.")

        ts = pd.to_datetime(df[self.cc.col_datetime], errors="coerce")
        mask = ~ts.isna()
        df = df.loc[mask].copy()
        df[self.cc.col_datetime] = ts[mask].values  # 인덱스 정합 보장

        if run_time is not None:
            rt = pd.to_datetime(run_time, errors="raise")
            # 만약 "YYYY-MM-DD" 같이 날짜만 들어오면, 같은 날 전체 포함되도록 EOD로 보정
            if isinstance(run_time, str) and len(run_time.strip()) == 10:
                rt = rt + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
            df = df.loc[df[self.cc.col_datetime] <= rt]
            
        if df.empty:
            when = f" <= {run_time}" if run_time else ""
            raise ValueError(f"No rows available for inference{when}.")

        # 가장 최근 1행만 반환
        last_ts = df[self.cc.col_datetime].max()
        df_last = df.loc[df[self.cc.col_datetime] == last_ts].head(1).copy()

        # 유효성(필요 컬럼 결측) 마스크: require_hz 옵션 반영
        valid_mask = self.select_valid_rows_for_inference(df_last, require_hz=require_hz)
        if not bool(valid_mask.iloc[0]):
            need_cols = (self._inference_required() if require_hz
                         else [self.cc.col_o2, self.cc.col_temp, self.cc.col_nox])
            raise ValueError(f"Latest row at {last_ts} has NA in required columns: {need_cols}")

        return df_last.reset_index(drop=True)

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터 유효성 요약 (학습/추론 공용)"""
        cc = self.cc
        train_req = self._training_required()
        infer_req = self._inference_required()  # feature + target

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

        return info

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        *,
        sample_size: Optional[int] = None,
        random_state: int = 42,
        dropna: bool = True,
        dedup: bool = True,
    ) -> pd.DataFrame:
        """
        학습 전처리:
        - 필요한 컬럼만 슬라이싱
        - 결측 제거/중복 제거
        - 샘플링(옵션)
        """
        req = self._training_required()
        d = df.copy()
        missing = [c for c in req if c not in d.columns]
        if missing:
            raise ValueError(f"Training data missing required columns: {missing}")

        d = d[req]
        if dropna:
            d = d.dropna(subset=req)
        if dedup:
            d = d.drop_duplicates(subset=req, keep="first")
        if sample_size is not None and len(d) > int(sample_size):
            d = d.sample(n=int(sample_size), random_state=int(random_state))
        return d.reset_index(drop=True)

    def select_valid_rows_for_inference(self, df: pd.DataFrame, *, require_hz: bool = False) -> pd.Series:
        """
        추론에 사용할 행(Boolean mask) 선택.
        - require_hz=False: O2, Temp(col_temp), target 만 필수(최적화 파이프라인과 호환)
        - require_hz=True : feature_columns + target 모두 필수 (직접 predict(X)용)
        """
        if require_hz:
            cols = self._inference_required()  # feature + target
        else:
            cols = [self.cc.col_o2, self.cc.col_temp, self.cc.col_nox]
        exist = [c for c in cols if c in df.columns]
        if not exist:
            raise ValueError(f"No required inference columns present: expected {cols}")
        return df[exist].notna().all(axis=1)

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _training_required(self) -> List[str]:
        return list(dict.fromkeys(self.cc.feature_columns + [self.cc.target_column]))

    def _training_columns(self) -> List[str]:
        cols = [self.cc.col_datetime] + self._training_required()
        cols += [self.cc.col_inner_temp, self.cc.col_outer_temp]
        return list(dict.fromkeys(cols))

    def _inference_required(self) -> List[str]:
        return list(dict.fromkeys(self.cc.feature_columns + [self.cc.target_column]))

    def _inference_columns(self) -> List[str]:
        cols = [self.cc.col_datetime] + self._inference_required()
        cols += [self.cc.col_inner_temp, self.cc.col_outer_temp]
        return list(dict.fromkeys(cols))

    def _load_any(self, path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".parquet", ".pq"):
            try:
                return pd.read_parquet(path, columns=columns)
            except Exception:
                df = pd.read_parquet(path)
                if columns:
                    cols_exist = [c for c in columns if c in df.columns]
                    return df[cols_exist]
                return df
        elif ext in (".csv", ".txt"):
            try:
                return pd.read_csv(path, usecols=columns)
            except Exception:
                df = pd.read_csv(path)
                if columns:
                    cols_exist = [c for c in columns if c in df.columns]
                    return df[cols_exist]
                return df
        elif ext in (".feather", ".ft"):
            df = pd.read_feather(path)
            if columns:
                cols_exist = [c for c in columns if c in df.columns]
                return df[cols_exist]
            return df
        else:
            raise ValueError(f"Unsupported data extension: {ext}")

    def _ensure_datetime(self, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        if time_col in df.columns:
            df = df.copy()
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        return df

    def _filter_by_time_range(
        self,
        df: pd.DataFrame,
        time_col: str,
        start_time: Optional[str],
        end_time: Optional[str],
    ) -> pd.DataFrame:
        if time_col not in df.columns:
            return df
        d = df.copy()
        if start_time:
            d = d[d[time_col] >= pd.to_datetime(start_time)]
        if end_time:
            d = d[d[time_col] <= pd.to_datetime(end_time)]
        return d.sort_values(time_col).reset_index(drop=True)
