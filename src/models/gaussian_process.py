# ======================
# gaussian_process.py
# ======================
"""
Gaussian Process Model for NOx prediction
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional

import os
import logging
from datetime import datetime, timezone, timedelta
from pprint import pformat

import numpy as np
import pandas as pd
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from config.column_config import ColumnConfig
from config.model_config import GPModelConfig

from utils.logger import get_logger  # LoggerConfig는 model_config가 들고 있음


@dataclass
class GaussianProcessNOxModel:
    """
    NOx 예측을 위한 Gaussian Process Regressor 래퍼
    - 컬럼 스키마: ColumnConfig
    - 모델/커널/학습/저장 파라미터: GPModelConfig
    """
    column_config: ColumnConfig = field(default_factory=ColumnConfig)
    model_config: GPModelConfig = field(default_factory=GPModelConfig)

    # 런타임
    model: Optional[GaussianProcessRegressor] = field(default=None, repr=False)
    # 클래스 필드에 추가
    _train_df_: Optional[pd.DataFrame] = field(default=None, repr=False)
    _n_train_: Optional[int] = None
    _trained_at_: Optional[str] = None  # 학습 완료 시각(KST), "YYYY-MM-DD HH:MM:SS"

    # 로거 인스턴스(설정은 model_config.logger_cfg에서 주입)
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = get_logger(self.model_config.logger_cfg)

    # ----------------------
    # 커널 생성 (내장: 시작 로그 생략)
    # ----------------------
    def _create_kernel(self) -> ConstantKernel:
        mc = self.model_config
        kernel = (
            ConstantKernel(mc.constant_value, mc.constant_bounds)
            * Matern(
                length_scale=np.array(mc.matern_length_scale_init, dtype=float),
                nu=mc.matern_nu
              )
            + WhiteKernel(
                noise_level=mc.white_noise_level,
                noise_level_bounds=mc.white_noise_bounds
              )
        )
        self.logger.debug(f"커널 생성: {kernel}")
        return kernel

    # ----------------------
    # 학습
    # ----------------------
    def fit(self, df: pd.DataFrame) -> "GaussianProcessNOxModel":
        """학습 데이터프레임으로 GP 모델 학습."""
        self.logger.info("GP Model 학습 시작")
        exec_file = os.path.basename(globals().get("__file__", "interactive"))
        self.logger.debug(f"실행 파일: {exec_file}")
        fmt = (lambda a:
               np.array2string(np.asarray(a), precision=6, separator=', ')
               if np.size(a) <= 10 else
               f"size={np.size(a)}, min={np.nanmin(a):.6g}, max={np.nanmax(a):.6g}, "
               f"head={np.asarray(a)[:2]}, tail={np.asarray(a)[-2:]}")

        cc, mc = self.column_config, self.model_config

        d = df.copy()
        # 배열 생성
        X = d[cc.gp_feature_columns].to_numpy(dtype=float)  # [Hz, O2, Temp(col_temp)]
        y = d[cc.target_column].to_numpy(dtype=float)
        
        # ▼ 최종 학습데이터(피처+타깃) 보관
        req_cols = list(dict.fromkeys(cc.gp_feature_columns + [cc.target_column]))
        self._train_df_ = d[req_cols].copy()
        self.logger.debug(f"학습 데이터 스냅샷 저장: shape={self._train_df_.shape}")


        # 간단 통계 로그
        self.logger.debug(f"X.shape={X.shape}, y.shape={y.shape}")
        for i, col in enumerate(cc.gp_feature_columns):
            colv = X[:, i]
            self.logger.debug(f"X[{col}] 통계:")
            self.logger.debug(f"count={colv.size}")
            self.logger.debug(f"mean ={np.nanmean(colv):.6g}")
            self.logger.debug(f"std  ={np.nanstd(colv):.6g}")
            self.logger.debug(f"min  ={np.nanmin(colv):.6g}")
            self.logger.debug(f"max  ={np.nanmax(colv):.6g}")
        self.logger.debug(f"y[{cc.target_column}] 통계:")
        self.logger.debug(f"count={y.size}")
        self.logger.debug(f"mean ={np.nanmean(y):.6g}")
        self.logger.debug(f"std  ={np.nanstd(y):.6g}")
        self.logger.debug(f"min  ={np.nanmin(y):.6g}")
        self.logger.debug(f"max  ={np.nanmax(y):.6g}")

        kernel = self._create_kernel()
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=mc.normalize_y,
            n_restarts_optimizer=mc.n_restarts_optimizer,
            random_state=mc.random_state,
            alpha=mc.alpha,
        )

        # 시작 파라미터 로그(기간 포함)
        self.logger.info(f"학습 파라미터:")
        self.logger.info(f"sample_n   ={len(d)}")
        self.logger.info(f"features   ={X.shape[1]}")
        self.logger.info(f"normalize_y={mc.normalize_y}")
        self.logger.info(f"n_restarts ={mc.n_restarts_optimizer}")
        self.logger.info(f"alpha      ={mc.alpha}")
        self.logger.info(f"기간       ={mc.start_time} ~ {mc.end_time}")

        self.model.fit(X, y)
        self._n_train_ = int(len(d))

        # 학습 시각(KST)
        _kst = timezone(timedelta(hours=9))
        self._trained_at_ = datetime.now(_kst).strftime("%Y-%m-%d %H:%M:%S")

        self.logger.info("GP Model 학습 완료")
        self.logger.debug(f"학습 완료 시각(KST): {self._trained_at_}")
        self.logger.debug(f"최적화된 커널: {self.model.kernel_}")
        return self

    # ----------------------
    # 예측
    # ----------------------
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """학습된 모델로 예측 수행."""
        self.logger.info("GP Model 예측 시작")
        exec_file = os.path.basename(globals().get("__file__", "interactive"))
        self.logger.debug(f"실행 파일: {exec_file}")
        fmt = (lambda a:
               np.array2string(np.asarray(a), precision=6, separator=', ')
               if np.size(a) <= 10 else
               f"size={np.size(a)}, min={np.nanmin(a):.6g}, max={np.nanmax(a):.6g}, "
               f"head={np.asarray(a)[:2]}, tail={np.asarray(a)[-2:]}")

        try:
            if self.model is None:
                msg = "모델이 학습되지 않았습니다. fit()을 먼저 호출하세요."
                self.logger.error(msg)
                raise ValueError(msg)

            if not isinstance(X, np.ndarray):
                msg = f"입력 타입 오류: numpy.ndarray 필요, 전달형={type(X)}"
                self.logger.error(msg)
                raise TypeError(msg)

            if X.ndim != 2 or X.shape[1] != 3:
                msg = f"입력 형상 오류: (n, 3) 필요, 전달형={X.shape}"
                self.logger.error(msg)
                raise ValueError(msg)

            if np.isnan(X).any():
                n_samples = X.shape[0]
                self.logger.warning("입력에 NaN 포함 → NaN 예측 반환")
                self.logger.debug(f"입력 X(요약): {fmt(X)}")
                return (np.full(n_samples, np.nan),
                        np.full(n_samples, np.nan) if return_std else None)

            self.logger.debug(f"입력 X.shape={X.shape}, return_std={return_std}")
            self.logger.debug(f"입력 X(요약): {fmt(X)}")

            pred_mean, pred_std = self.model.predict(X, return_std=return_std)

            # 출력 요약
            self.logger.debug(f"예측 μ 요약: {fmt(pred_mean)}")
            if return_std:
                self.logger.debug(f"예측 σ 요약: {fmt(pred_std)}")

            self.logger.info("GP Model 예측 완료")
            return pred_mean, pred_std

        except:
            self.logger.exception("predict() 중 에러 발생 -> NaN으로 예외 처리")


    # ----------------------
    # 저장
    # ----------------------
    def save(self, filepath: Optional[str] = None) -> str:
        """모델 저장. filepath가 없으면 GPModelConfig.model_path 사용."""
        # 시작 로그 + 실행 파일
        self.logger.info("GP Model 저장 시작")
        exec_file = os.path.basename(globals().get("__file__", "interactive"))
        self.logger.debug(f"실행 파일: {exec_file}")
    
        if self.model is None:
            msg = "모델이 학습되지 않았습니다."
            self.logger.error(msg)
            raise ValueError(msg)
    
        path = filepath or self.model_config.model_path
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
    
        # 1) 모델 저장
        joblib.dump(self.model, path)
        self.logger.info(f"모델 저장 완료: {path}")
    
        # 2) 학습 데이터 CSV 저장 (가능한 경우)
        if self._train_df_ is None:
            self.logger.warning("학습 데이터 스냅샷이 없어 CSV 저장을 건너뜁니다.")
        else:
            base = os.path.splitext(os.path.basename(path))[0]
            csv_path = os.path.join(dir_ if dir_ else ".", f"{base}_train.csv")
            self._train_df_.to_csv(csv_path, index=False, encoding="utf-8-sig")
            self.logger.info(
                f"학습 데이터 CSV 저장 완료: {csv_path} "
                f"(rows={len(self._train_df_)}, cols={list(self._train_df_.columns)})"
            )
    
        return path


    # ----------------------
    # 로드
    # ----------------------
    def load(self, filepath: Optional[str] = None) -> "GaussianProcessNOxModel":
        """모델 로드. filepath가 없으면 GPModelConfig.model_path 사용."""
        self.logger.info("GP Model 로드 시작")
        exec_file = os.path.basename(globals().get("__file__", "interactive"))
        self.logger.debug(f"실행 파일: {exec_file}")

        path = filepath or self.model_config.model_path
        if not os.path.exists(path):
            msg = f"모델 파일을 찾을 수 없습니다: {path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)

        self.model = joblib.load(path)
        self.logger.info(f"GP Model 로드 완료: {path}")
        return self

    # ----------------------
    # 메타 정보
    # ----------------------
    def get_model_info(self) -> dict:
        """모델 메타 정보 조회."""
        self.logger.info("GP Model 정보 조회 시작")
        exec_file = os.path.basename(globals().get("__file__", "interactive"))
        self.logger.debug(f"실행 파일: {exec_file}")

        if self.model is None:
            self.logger.warning("모델 정보 요청: 학습 전 상태(not_trained)")
            return {"status": "not_trained"}

        info = {
            "status": "trained",
            "kernel": str(self.model.kernel_),
            "n_features": int(self.model.n_features_in_),
            "n_train": self._n_train_,
            "random_state": self.model_config.random_state,
            "trained_at": self._trained_at_,  # 학습 완료 시각(KST)
            "train_period": {
                "start": self.model_config.start_time,
                "end": self.model_config.end_time,
            },
        }
        self.logger.debug("모델 정보:")
        text = pformat(info, width=120, compact=False)
        for line in text.splitlines():
            self.logger.info(line)
        self.logger.info("GP Model 정보 조회 완료")
        return info