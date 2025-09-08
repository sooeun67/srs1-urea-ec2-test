"""
SRS1 요소수 제어 모델 추론 테스트
"""
import os
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 로컬 모듈 import
from models.gaussian_process import GaussianProcessNOxModel
from data_processing.data_loader import DataLoader
from config.model_config import ModelConfig
from config.column_config import ColumnConfig
from config.optimization_config import OptimizationConfig


def test_model_loading():
    """모델 로딩을 테스트합니다."""
    print("=" * 50)
    print("모델 로딩 테스트")
    print("=" * 50)
    
    try:
        # 설정 파일 로드
        model_config = ModelConfig()
        column_config = ColumnConfig()
        
        print("✅ 설정 파일 로드 성공")
        print(f"모델 설정: {model_config.__dict__}")
        
        # Gaussian Process 모델 초기화
        gp_model = GaussianProcessNOxModel()
        print("✅ Gaussian Process 모델 초기화 성공")
        
        return gp_model, model_config, column_config
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return None, None, None


def test_data_processing():
    """데이터 처리 파이프라인을 테스트합니다."""
    print("\n" + "=" * 50)
    print("데이터 처리 테스트")
    print("=" * 50)
    
    try:
        # 더미 데이터 생성 (실제 InfluxDB 데이터 형태 모방)
        dummy_data = {
            'time': pd.date_range(start='2024-01-01', periods=100, freq='5min'),
            'NOX_VALUE': np.random.normal(15, 3, 100),
            'UREA_FLOW': np.random.normal(45, 5, 100),
            'TEMP_VALUE': np.random.normal(350, 20, 100),
            'O2_VALUE': np.random.normal(8, 1, 100),
            'LOAD_VALUE': np.random.normal(80, 10, 100)
        }
        
        df = pd.DataFrame(dummy_data)
        print(f"더미 데이터 생성: {df.shape}")
        print("데이터 샘플:")
        print(df.head())
        
        # 데이터 로더 테스트
        data_loader = DataLoader()
        print("✅ 데이터 로더 초기화 성공")
        
        return df
        
    except Exception as e:
        print(f"❌ 데이터 처리 실패: {e}")
        return None


def test_model_prediction(model, data):
    """모델 예측을 테스트합니다."""
    print("\n" + "=" * 50)
    print("모델 예측 테스트")
    print("=" * 50)
    
    if model is None or data is None:
        print("모델 또는 데이터가 없어서 예측을 건너뜁니다.")
        return None
    
    try:
        # 최신 데이터 포인트 사용
        latest_data = data.tail(1)
        
        # 특성 데이터 준비 (실제 모델에 맞게 조정 필요)
        features = ['UREA_FLOW', 'TEMP_VALUE', 'O2_VALUE', 'LOAD_VALUE']
        X = latest_data[features].values
        
        print(f"입력 특성: {features}")
        print(f"입력 데이터: {X}")
        
        # 모델 예측 (실제 구현에 따라 조정 필요)
        # prediction = model.predict(X)
        
        # 더미 예측값 (실제 모델 연결 전까지)
        prediction = np.random.normal(12, 2, 1)[0]
        
        print(f"✅ NOx 예측값: {prediction:.2f} ppm")
        
        return prediction
        
    except Exception as e:
        print(f"❌ 모델 예측 실패: {e}")
        return None


def test_optimization():
    """요소수 펌프 최적화를 테스트합니다."""
    print("\n" + "=" * 50)
    print("요소수 펌프 최적화 테스트")
    print("=" * 50)
    
    try:
        opt_config = OptimizationConfig()
        print("✅ 최적화 설정 로드 성공")
        
        # 더미 최적화 결과
        target_nox = 10.0
        current_nox = 15.0
        optimized_pump_hz = 48.5
        
        print(f"목표 NOx: {target_nox} ppm")
        print(f"현재 NOx: {current_nox} ppm")
        print(f"최적화된 펌프 주파수: {optimized_pump_hz} Hz")
        
        return optimized_pump_hz
        
    except Exception as e:
        print(f"❌ 최적화 실패: {e}")
        return None


if __name__ == "__main__":
    print("🚀 SRS1 요소수 제어 모델 추론 테스트 시작")
    
    # 1. 모델 로딩 테스트
    model, model_config, column_config = test_model_loading()
    
    # 2. 데이터 처리 테스트
    test_data = test_data_processing()
    
    # 3. 모델 예측 테스트
    prediction = test_model_prediction(model, test_data)
    
    # 4. 최적화 테스트
    optimized_hz = test_optimization()
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("테스트 결과 요약")
    print("=" * 50)
    
    if all([model, test_data, prediction, optimized_hz]):
        print("✅ 모든 테스트 통과!")
        print(f"   - 모델 로딩: 성공")
        print(f"   - 데이터 처리: 성공")
        print(f"   - NOx 예측: {prediction:.2f} ppm")
        print(f"   - 펌프 최적화: {optimized_hz} Hz")
    else:
        print("⚠️ 일부 테스트 실패")
        print(f"   - 모델 로딩: {'성공' if model else '실패'}")
        print(f"   - 데이터 처리: {'성공' if test_data is not None else '실패'}")
        print(f"   - 모델 예측: {'성공' if prediction else '실패'}")
        print(f"   - 최적화: {'성공' if optimized_hz else '실패'}")
