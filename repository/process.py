import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.config import ProcessConfig
from utils.logger import set_logger

set_logger(__name__)
custom_logger = logging.getLogger(__name__)


class ProcessRepo:
    def __init__(
        self,
        process_config: Optional[ProcessConfig] = None,
    ) -> None:
        """Initialize ProcessRepo for urea control system data preprocessing

        Parameters
        ----------
        process_config : Optional[ProcessConfig], optional, by default None
        """
        config = process_config or ProcessConfig()
        
        # Column information for urea control system
        self.feature_columns = config.feature_columns  # ["snr_pmp_uw_s_1", "br1_eo_o2_a", "icf_ccs_fg_t_1"]
        self.target_column = config.target_column      # "icf_tms_nox_a"
        self.datetime_column = config.datetime_column  # "_time_gateway"

        # Data quality configurations
        self.defect_ratio = config.defect_ratio
        self.min_temp = config.min_temp
        self.max_temp = config.max_temp
        self.min_o2 = config.min_o2
        self.max_o2 = config.max_o2
        self.min_nox = config.min_nox
        self.max_nox = config.max_nox
        self.min_hz = config.min_hz
        self.max_hz = config.max_hz
        
        # Column mapping for data quality checks (influx.py와 연동)
        self.column_mapping = {
            'temp': 'icf_ccs_fg_t_1',           # 소각로 내부 온도
            'outer_temp': 'icf_scs_fg_t_1',     # 소각로 출구 온도
            'o2': 'br1_eo_o2_a',                # 보일러 출구 산소 농도
            'nox_bf': 'icf_tms_nox_a',          # NOx (보정 전)
            'hz': 'snr_pmp_uw_s_1',             # 요소수 펌프 Hz
        }

    def set_time_index(self, X: pd.DataFrame) -> pd.DataFrame:
        """Set time index for urea control data from influx.py
        
        influx.py에서 받은 5초 윈도우 데이터의 시간 인덱스를 설정합니다.

        Parameters
        ----------
        X : pd.DataFrame
            influx.py에서 받은 센서 데이터 (dict 형태)

        Returns
        -------
        pd.DataFrame
            시간 인덱스가 설정된 DataFrame
        """
        # dict를 DataFrame으로 변환 (단일 행)
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        
        # _time_gateway를 인덱스로 설정
        if self.datetime_column in X.columns:
            X = X.set_index(self.datetime_column)
            X.index = pd.to_datetime(X.index)
        else:
            custom_logger.warning(f"Column {self.datetime_column} not found, using default index")
        
        return X

    def check_data_quality(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Check urea control data quality for model inference
        
        요소수 제어 모델 추론을 위한 데이터 품질을 검사합니다.

        Parameters
        ----------
        X : pd.DataFrame
            검사할 데이터

        Returns
        -------
        Tuple[pd.DataFrame, bool]
            검사된 데이터와 품질 문제 여부
        """
        is_data_invalid = False
        
        # Check for missing values
        missing_ratio = X.isna().sum(axis=0) / X.shape[0]
        if (missing_ratio > self.defect_ratio).any():
            custom_logger.warning(
                f"Some features contain over {self.defect_ratio * 100}% missing value.\n"
                f"{missing_ratio[missing_ratio > self.defect_ratio].to_markdown(tablefmt='psql')}"
            )
            is_data_invalid = True

        # Check temperature range (소각로 내부 온도)
        temp_col = self.column_mapping.get('temp', 'icf_ccs_fg_t_1')
        if temp_col in X.columns:
            temp_outliers = (X[temp_col] < self.min_temp) | (X[temp_col] > self.max_temp)
            if temp_outliers.any():
                custom_logger.warning(f"Temperature outliers detected in {temp_col}: {temp_outliers.sum()} points")
                is_data_invalid = True

        # Check O2 range (보일러 출구 산소 농도)
        o2_col = self.column_mapping.get('o2', 'br1_eo_o2_a')
        if o2_col in X.columns:
            o2_outliers = (X[o2_col] < self.min_o2) | (X[o2_col] > self.max_o2)
            if o2_outliers.any():
                custom_logger.warning(f"O2 outliers detected in {o2_col}: {o2_outliers.sum()} points")
                is_data_invalid = True

        # Check NOx range (NOx 보정 전)
        nox_col = self.column_mapping.get('nox_bf', 'icf_tms_nox_a')
        if nox_col in X.columns:
            nox_outliers = (X[nox_col] < self.min_nox) | (X[nox_col] > self.max_nox)
            if nox_outliers.any():
                custom_logger.warning(f"NOx outliers detected in {nox_col}: {nox_outliers.sum()} points")
                is_data_invalid = True

        # Check Hz range (요소수 펌프 Hz)
        hz_col = self.column_mapping.get('hz', 'snr_pmp_uw_s_1')
        if hz_col in X.columns:
            hz_outliers = (X[hz_col] < self.min_hz) | (X[hz_col] > self.max_hz)
            if hz_outliers.any():
                custom_logger.warning(f"Hz outliers detected in {hz_col}: {hz_outliers.sum()} points")
                is_data_invalid = True

        return X, is_data_invalid

    def apply_nan_fill(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values with linear interpolation for urea control data
        
        influx.py에서 받은 5초 윈도우 데이터의 NaN 값을 선형 보간으로 채웁니다.

        Parameters
        ----------
        X : pd.DataFrame
            NaN 값이 포함된 데이터

        Returns
        -------
        pd.DataFrame
            NaN 값이 채워진 데이터
        """
        # Feature columns: Fill NaN values by linear interpolation
        feature_columns = X.columns.intersection(self.feature_columns).to_list()
        if feature_columns:
            X.loc[:, feature_columns] = X.loc[:, feature_columns].interpolate(
                method="linear",
                limit_direction="both",
                axis=0,
            )

        # When all values were NaN before interpolation, they still remain as NaN
        if nan_columns := X.columns[X.isna().any()].intersection(feature_columns).to_list():
            custom_logger.warning(
                f"Remaining NaNs detected in columns: {nan_columns}. "
                "These will be handled by the model scaler."
            )

        return X

    def prepare_urea_control_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for urea control model inference
        
        influx.py에서 받은 5초 윈도우 데이터를 모델 추론용 피처로 준비합니다.
        이 함수가 요소수 제어 모델의 핵심 전처리 함수입니다.

        Parameters
        ----------
        X : pd.DataFrame
            influx.py에서 받은 센서 데이터

        Returns
        -------
        pd.DataFrame
            모델 추론용으로 준비된 피처 데이터
        """
        # 1. 시간 인덱스 설정
        X = self.set_time_index(X)
        
        # 2. 데이터 품질 검사
        X, is_invalid = self.check_data_quality(X)
        
        if is_invalid:
            custom_logger.warning("Data quality issues detected, but continuing with preprocessing")
        
        # 3. NaN 값 채우기
        X = self.apply_nan_fill(X)
        
        # 4. 피처 컬럼만 선택하여 반환
        feature_data = X[self.feature_columns].copy()
        
        custom_logger.info(f"Prepared {len(feature_data)} feature columns for urea control model")
        return feature_data