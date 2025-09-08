import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import mlflow

from utils.config import ModelConfig
from utils.logger import set_logger

set_logger(__name__)
custom_logger = logging.getLogger(__name__)


class ModelRepository:
    """MLflow 모델 다운로드 및 관리 클래스 - 요소수 제어 시스템용"""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """ModelRepository 초기화
        
        Args:
            tracking_uri: MLflow tracking URI. None이면 환경변수 MLFLOW_TRACKING_URI 사용
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        self.model_cache_dir = Path("/tmp/mlflow_models")
        self.model_cache_dir.mkdir(exist_ok=True)
    
    def download_model_from_run_id(
        self, 
        run_id: str, 
        model_name: str = "urea_gp_model",
        target_path: Optional[Path] = None
    ) -> Path:
        """MLflow run_id로부터 요소수 제어 모델 다운로드
        
        Args:
            run_id: MLflow run ID
            model_name: 모델 아티팩트 이름 (기본값: "urea_gp_model")
            target_path: 저장할 경로. None이면 캐시 디렉토리 사용
            
        Returns:
            다운로드된 모델 경로
        """
        try:
            if target_path is None:
                target_path = self.model_cache_dir / f"{run_id}_{model_name}"
            
            custom_logger.info(f"Downloading urea control model from run_id: {run_id}")
            
            # MLflow에서 모델 다운로드
            mlflow.artifacts.download_artifacts(
                artifact_uri=f"runs:/{run_id}/{model_name}",
                dst_path=str(target_path)
            )
            
            custom_logger.info(f"Urea control model downloaded successfully to: {target_path}")
            return target_path
            
        except Exception as e:
            custom_logger.error(f"Failed to download urea control model from run_id {run_id}: {str(e)}")
            raise
    
    def get_latest_urea_model_run_id(
        self, 
        experiment_name: str, 
        filter_string: Optional[str] = None
    ) -> str:
        """요소수 제어 실험에서 최신 모델의 run_id 조회
        
        Args:
            experiment_name: MLflow 실험 이름
            filter_string: 필터 조건 (예: "tags.model_type='urea_gp'")
            
        Returns:
            최신 run_id
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_name}' not found")
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs.empty:
                raise ValueError(f"No runs found in experiment '{experiment_name}'")
            
            run_id = runs.iloc[0]["run_id"]
            custom_logger.info(f"Latest urea control model run_id found: {run_id}")
            return run_id
            
        except Exception as e:
            custom_logger.error(f"Failed to get latest urea control model run_id: {str(e)}")
            raise
    
    def download_latest_urea_model(
        self, 
        experiment_name: str, 
        model_name: str = "urea_gp_model",
        filter_string: Optional[str] = None
    ) -> Path:
        """최신 요소수 제어 모델 다운로드
        
        Args:
            experiment_name: MLflow 실험 이름
            model_name: 모델 아티팩트 이름
            filter_string: 필터 조건
            
        Returns:
            다운로드된 모델 경로
        """
        run_id = self.get_latest_urea_model_run_id(experiment_name, filter_string)
        return self.download_model_from_run_id(run_id, model_name)
    
    def cleanup_old_models(self, keep_count: int = 5):
        """오래된 모델 캐시 정리
        
        Args:
            keep_count: 유지할 모델 개수
        """
        try:
            model_dirs = sorted(
                [d for d in self.model_cache_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            for old_dir in model_dirs[keep_count:]:
                import shutil
                shutil.rmtree(old_dir)
                custom_logger.info(f"Cleaned up old urea control model cache: {old_dir}")
                
        except Exception as e:
            custom_logger.warning(f"Failed to cleanup old urea control models: {str(e)}")


class UreaModelRepo:
    """요소수 제어 모델 관리 클래스 - Gaussian Process 모델 전용"""

    def __init__(self, model_config: Optional[ModelConfig] = None) -> None:
        """initialize UreaModelRepo with given model config

        Parameters
        ----------
        model_config : Optional[ModelConfig], optional, by default None
        """

        self._config = model_config or ModelConfig()

        # The model directory from ModelConfig (e.g., "/opt/ml/model" in Lambda)
        self.model_dir = Path(self._config.model_dir)

        # Gaussian Process model paths using config file names
        self.gp_model_path = self.model_dir / self._config.gp_model_file
        self.gp_config_path = self.model_dir / self._config.config_file

    @property
    def gp_model(self):
        """Return the Gaussian Process model itself as a property.

        Returns
        -------
        GaussianProcessNOxModel
        """
        # When GP model hasn't been loaded
        if not hasattr(self, "_gp_model"):
            custom_logger.info(f"Loading Gaussian Process model from '{self.gp_model_path}'")
            import joblib
            self._gp_model = joblib.load(self.gp_model_path)
        return self._gp_model

    @property
    def gp_config(self) -> Dict[str, Any]:
        """Return the configurations of the Gaussian Process model.

        Returns
        -------
        Dict[str, Any]
        """
        # When GP config hasn't been loaded
        if not hasattr(self, "_gp_config"):
            custom_logger.info(f"Loading Gaussian Process model configurations from '{self.gp_config_path}'")
            with open(self.gp_config_path, "r", encoding="utf-8") as gp_config_yaml:
                self._gp_config = yaml.full_load(gp_config_yaml)
        return self._gp_config

    def get_column_info(self) -> Dict[str, List[str]]:
        """Return the feature and target column information from the GP model.
        The usage of this method is when updating the column info of `ProcessRepo` with the column
        info of `UreaModelRepo`. This method make the `InferenceService` available to be initialized without
        loading the GP model.
        """
        return {
            "feature_columns": self.gp_config["data_config"]["feature_columns"],
            "target_column": self.gp_config["data_config"]["target_column"],
        }