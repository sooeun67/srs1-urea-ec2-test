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
import numpy as np
import pandas as pd
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from config.column_config import ColumnConfig
from config.model_config import ModelConfig

@dataclass
class GaussianProcessNOxModel:
    """
    NOx 예측을 위한 Gaussian Process Regressor 래퍼
    - 컬럼 스키마: ColumnConfig
    - 모델/커널/학습/저장 파라미터: ModelConfig
    """
    column_config: ColumnConfig = field(default_factory=ColumnConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)

    # 런타임
    model: Optional[GaussianProcessRegressor] = field(default=None, repr=False)
    _n_train_: Optional[int] = None

    # ----------------------
    # 커널 생성
    # ----------------------
    def _create_kernel(self) -> ConstantKernel:
        mc = self.model_config
        return (
            ConstantKernel(mc.constant_value, mc.constant_bounds)
            * Matern(length_scale=np.array(mc.matern_length_scale_init, dtype=float),
                     nu=mc.matern_nu)
            + WhiteKernel(noise_level=mc.white_noise_level,
                          noise_level_bounds=mc.white_noise_bounds)
        )

    # ----------------------
    # 학습
    # ----------------------
    def fit(self, df: pd.DataFrame) -> "GaussianProcessNOxModel":
        """
        df : 학습 데이터프레임. 반드시 ColumnConfig의
             feature_columns + [target_column] 포함.
        """
        cc, mc = self.column_config, self.model_config

        req_cols = list(dict.fromkeys(cc.feature_columns + [cc.target_column]))
        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            raise ValueError(f"df must include columns: {req_cols} (missing: {missing})")

        d = df[req_cols].copy()

        # if mc.dropna_required:
        #     d = d.dropna(subset=req_cols)

        if len(d) < mc.min_samples:
            raise ValueError(f"학습 데이터가 너무 적습니다(최소 {mc.min_samples}행 권장).")

        if mc.dedup:
            d = d.drop_duplicates(subset=req_cols, keep="first")

        if mc.sample_size is not None and len(d) > mc.sample_size:
            d = d.sample(n=int(mc.sample_size), random_state=mc.random_state)
            
        if mc.dropna_required:
            d = d.dropna(subset=req_cols)

        X = d[cc.feature_columns].to_numpy(dtype=float)  # [Hz, O2, Temp(col_temp)]
        y = d[cc.target_column].to_numpy(dtype=float)

        kernel = self._create_kernel()
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=mc.normalize_y,
            n_restarts_optimizer=mc.n_restarts_optimizer,
            random_state=mc.random_state,
            alpha=mc.alpha,
        )
        self.model.fit(X, y)
        self._n_train_ = int(len(d))
        return self

    # ----------------------
    # 예측
    # ----------------------
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        학습된 Gaussian Process 모델로 예측 수행.
        
        Args:
            X: (n_samples, 3) = [Hz, O2, Temp(col_temp)]
            return_std: 표준편차 반환 여부 (기본: True)

        Returns:
            Tuple[pred_mean, pred_std]: 결측 시 NaN 반환
        """
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        # 입력 유효성 체크
        if not isinstance(X, np.ndarray):
            raise TypeError("X는 numpy.ndarray 형태여야 합니다.")

        # 하나라도 NaN이 있으면 메시지 출력 + NaN 결과 반환
        if np.isnan(X).any():
            print("[WARN] 입력값에 결측치가 있습니다. 예측값을 NaN으로 반환합니다.")
            n_samples = X.shape[0] if X.ndim == 2 else 1
            return (np.full(n_samples, np.nan), np.full(n_samples, np.nan) if return_std else None)

        # 결측 없으면 정상 예측
        return self.model.predict(X, return_std=return_std)

    # ----------------------
    # 저장/로드
    # ----------------------
    def save(self, filepath: Optional[str] = None) -> str:
        """모델 저장. filepath가 없으면 ModelConfig.model_path 사용."""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        path = filepath or self.model_config.model_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        return path

    def load(self, filepath: Optional[str] = None) -> "GaussianProcessNOxModel":
        """모델 로드. filepath가 없으면 ModelConfig.model_path 사용."""
        path = filepath or self.model_config.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        self.model = joblib.load(path)
        return self

    # ----------------------
    # 메타 정보
    # ----------------------
    def get_model_info(self) -> dict:
        if self.model is None:
            return {"status": "not_trained"}
        return {
            "status": "trained",
            "kernel": str(self.model.kernel_),
            "n_features": int(self.model.n_features_in_),
            "n_train": self._n_train_,
            "random_state": self.model_config.random_state,
        }
