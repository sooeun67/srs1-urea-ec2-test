# SRS1 요소수 제어 모델 - EC2 테스트 환경

## 📋 개요
이 폴더는 EC2 환경에서 SRS1 요소수 제어 모델의 실시간 데이터 처리 및 추론 기능을 테스트하기 위한 간소화된 코드입니다.

## 📁 폴더 구조
```
SRS1-ec2-test/
├── models/                    # Gaussian Process 모델
├── data_processing/           # 데이터 로더 및 전처리
├── config/                    # 설정 파일들
├── test_scripts/              # EC2 테스트 스크립트
├── requirements-ec2.txt       # EC2용 간소화된 requirements
└── README-EC2.md             # 이 파일
```

## 🚀 EC2 설정 및 실행 가이드

### 1. 자동 환경 설정 (권장)
```bash
# [0908] SRS1-sooeun과 동일한 환경을 EC2에 구성
chmod +x setup_ec2_environment.sh
./setup_ec2_environment.sh

# 환경 활성화
source activate_urea_env.sh
```

### 2. 수동 환경 설정 (선택사항)
```bash
# Conda 환경 생성 (environment.yml 사용)
conda env create -f environment.yml
conda activate urea-srs1

# 또는 requirements로 설치
conda create -n urea-srs1 python=3.11 -y
conda activate urea-srs1
pip install -r requirements-ec2.txt
```

### 2. 테스트 실행
```bash
# InfluxDB 연결 테스트
python test_scripts/test_influxdb_connection.py

# 실시간 데이터 조회 테스트
python test_scripts/test_realtime_data.py

# 모델 추론 테스트
python test_scripts/test_model_inference.py
```

## 🔗 관련 링크
- **서비스 코드**: `/home/insight-ai-02/skep-urea/SRS1-sooeun/`
- **GitHub**: https://github.com/sooeun67/nox
- **MLflow**: http://10.250.109.206:5000

## 📝 테스트 목표
1. InfluxDB 실시간 데이터 연결 확인
2. 데이터 전처리 파이프라인 검증
3. Gaussian Process 모델 추론 성능 확인
4. MLflow 모델 로딩 테스트

## ⚠️ 주의사항
- 이 코드는 테스트 목적으로만 사용
- 프로덕션 배포는 `/SRS1-sooeun/` 사용
- 테스트 결과를 바탕으로 서비스 코드 개선
