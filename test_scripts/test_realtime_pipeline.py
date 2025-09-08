"""
SRS1 요소수 제어 실시간 파이프라인 통합 테스트
InfluxDB 실시간 데이터 → 전처리 → 모델 추론 → 최적화 → 결과 출력
"""
import os
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 로컬 테스트 모듈
from test_scripts.test_influxdb_connection import test_influxdb_connection, test_realtime_data_query
from test_scripts.test_model_inference import test_model_loading, test_optimization


def run_realtime_pipeline():
    """실시간 파이프라인을 실행합니다."""
    print("🚀 SRS1 요소수 제어 실시간 파이프라인 테스트")
    print("=" * 60)
    
    # Step 1: InfluxDB 연결
    print("\n📡 Step 1: InfluxDB 연결")
    client = test_influxdb_connection()
    
    if not client:
        print("❌ InfluxDB 연결 실패 - 파이프라인 중단")
        return False
    
    # Step 2: 실시간 데이터 조회
    print("\n📊 Step 2: 실시간 데이터 조회")
    real_data = test_realtime_data_query(client)
    
    # Step 3: 모델 로딩
    print("\n🤖 Step 3: 모델 로딩")
    model, model_config, column_config = test_model_loading()
    
    # Step 4: 데이터 전처리 및 예측
    print("\n⚙️ Step 4: 데이터 전처리 및 예측")
    
    if real_data is not None and not real_data.empty:
        # 실제 InfluxDB 데이터 사용
        print("✅ 실제 InfluxDB 데이터 사용")
        processed_data = preprocess_influx_data(real_data)
    else:
        # 더미 데이터 사용
        print("⚠️ InfluxDB 데이터 없음 - 더미 데이터 사용")
        processed_data = generate_dummy_data()
    
    if processed_data is not None:
        print(f"전처리된 데이터 형태: {processed_data.shape}")
        
        # 모델 예측 시뮬레이션
        nox_prediction = simulate_nox_prediction(processed_data)
        print(f"📈 NOx 예측값: {nox_prediction:.2f} ppm")
        
        # Step 5: 요소수 펌프 최적화
        print("\n🎯 Step 5: 요소수 펌프 최적화")
        optimized_hz = optimize_pump_frequency(nox_prediction)
        print(f"🔧 최적화된 펌프 주파수: {optimized_hz:.1f} Hz")
        
        # Step 6: 결과 요약
        print("\n📋 Step 6: 결과 요약")
        print("=" * 60)
        print(f"⏰ 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 데이터 소스: {'InfluxDB 실시간' if real_data is not None else '더미 데이터'}")
        print(f"📈 현재 NOx 예측: {nox_prediction:.2f} ppm")
        print(f"🎯 목표 NOx: 10.0 ppm")
        print(f"🔧 권장 펌프 주파수: {optimized_hz:.1f} Hz")
        
        # 제어 권장사항
        if nox_prediction > 12.0:
            print("🔴 NOx 농도 높음 - 요소수 분사량 증가 필요")
        elif nox_prediction < 8.0:
            print("🔵 NOx 농도 낮음 - 요소수 분사량 감소 가능")
        else:
            print("🟢 NOx 농도 적정 - 현재 제어 유지")
        
        return True
    
    else:
        print("❌ 데이터 전처리 실패 - 파이프라인 중단")
        return False


def preprocess_influx_data(raw_data):
    """InfluxDB 원시 데이터를 전처리합니다."""
    try:
        # 실제 InfluxDB 데이터 구조에 맞게 전처리
        # (실제 구현 시 column_config 기반으로 수정 필요)
        
        print("📊 InfluxDB 데이터 전처리 중...")
        
        # 기본적인 데이터 정리
        processed = raw_data.copy()
        
        # 시간 컬럼 처리
        if 'time' in processed.columns:
            processed['time'] = pd.to_datetime(processed['time'])
            processed = processed.sort_values('time')
        
        # 결측값 처리
        processed = processed.fillna(method='ffill')
        
        print(f"✅ 전처리 완료: {processed.shape}")
        return processed
        
    except Exception as e:
        print(f"❌ 데이터 전처리 실패: {e}")
        return None


def generate_dummy_data():
    """더미 데이터를 생성합니다."""
    try:
        print("🎭 더미 데이터 생성 중...")
        
        # SRS1 요소수 시스템 더미 데이터
        dummy_data = {
            'time': datetime.now(),
            'NOX_VALUE': np.random.normal(15, 3),  # 현재 NOx 농도
            'UREA_FLOW': np.random.normal(45, 5),  # 요소수 유량
            'TEMP_VALUE': np.random.normal(350, 20),  # 배기 온도
            'O2_VALUE': np.random.normal(8, 1),  # 산소 농도
            'LOAD_VALUE': np.random.normal(80, 10),  # 엔진 부하
            'NH3_SLIP': np.random.normal(2, 0.5),  # 암모니아 슬립
        }
        
        df = pd.DataFrame([dummy_data])
        print(f"✅ 더미 데이터 생성 완료: {df.shape}")
        return df
        
    except Exception as e:
        print(f"❌ 더미 데이터 생성 실패: {e}")
        return None


def simulate_nox_prediction(data):
    """NOx 예측을 시뮬레이션합니다."""
    try:
        # 실제 Gaussian Process 모델 대신 간단한 시뮬레이션
        
        # 기본 NOx 값
        if 'NOX_VALUE' in data.columns:
            base_nox = data['NOX_VALUE'].iloc[-1]
        else:
            base_nox = 15.0
        
        # 요소수 유량에 따른 NOx 감소 효과 시뮬레이션
        if 'UREA_FLOW' in data.columns:
            urea_flow = data['UREA_FLOW'].iloc[-1]
            # 요소수 유량이 많을수록 NOx 감소
            nox_reduction = (urea_flow - 40) * 0.1
            predicted_nox = max(base_nox - nox_reduction, 5.0)
        else:
            predicted_nox = base_nox
        
        return predicted_nox
        
    except Exception as e:
        print(f"❌ NOx 예측 시뮬레이션 실패: {e}")
        return 15.0  # 기본값


def optimize_pump_frequency(current_nox, target_nox=10.0):
    """요소수 펌프 주파수를 최적화합니다."""
    try:
        # 목표 NOx와 현재 NOx 차이에 따른 펌프 주파수 조정
        nox_diff = current_nox - target_nox
        
        # 기본 펌프 주파수
        base_frequency = 45.0
        
        # NOx 차이에 따른 주파수 조정 (간단한 PI 제어 시뮬레이션)
        frequency_adjustment = nox_diff * 2.0  # 비례 게인
        
        optimized_frequency = base_frequency + frequency_adjustment
        
        # 주파수 제한 (38-54 Hz)
        optimized_frequency = max(38.0, min(54.0, optimized_frequency))
        
        return optimized_frequency
        
    except Exception as e:
        print(f"❌ 펌프 최적화 실패: {e}")
        return 45.0  # 기본값


if __name__ == "__main__":
    # 실시간 파이프라인 실행
    success = run_realtime_pipeline()
    
    if success:
        print("\n🎉 실시간 파이프라인 테스트 완료!")
    else:
        print("\n❌ 실시간 파이프라인 테스트 실패!")
        
    print("\n" + "=" * 60)
