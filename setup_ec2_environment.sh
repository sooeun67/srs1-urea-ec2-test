#!/bin/bash

# SRS1 요소수 제어 모델 - EC2 환경 설정 스크립트
# [0908] SRS1-sooeun과 동일한 가상환경을 EC2에 구성

set -e  # 에러 발생 시 스크립트 중단

# 색상 출력
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 SRS1 요소수 제어 모델 - EC2 환경 설정${NC}"
echo "=================================================="
echo -e "${BLUE}📋 목적: SRS1-sooeun과 동일한 가상환경 구성${NC}"
echo -e "${BLUE}🎯 대상: EC2 인스턴스 (x86_64)${NC}"
echo -e "${BLUE}🐍 Python: 3.11${NC}"
echo "=================================================="

# 1. 시스템 업데이트
echo -e "${YELLOW}📦 시스템 패키지 업데이트${NC}"
sudo apt-get update -y
sudo apt-get install -y curl wget git build-essential

# 2. Miniconda 설치 확인
echo -e "${YELLOW}🐍 Conda 설치 확인${NC}"
if ! command -v conda &> /dev/null; then
    echo "Conda가 설치되지 않았습니다. Miniconda 설치 중..."
    
    # Miniconda 다운로드 및 설치
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    
    # PATH에 추가
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    
    # 설치 파일 정리
    rm miniconda.sh
    
    echo -e "${GREEN}✅ Miniconda 설치 완료${NC}"
else
    echo -e "${GREEN}✅ Conda 이미 설치됨${NC}"
fi

# 3. Conda 초기화
echo -e "${YELLOW}🔧 Conda 초기화${NC}"
source $HOME/miniconda3/bin/activate
conda init bash

# 4. 가상환경 생성
echo -e "${YELLOW}🏗️ 가상환경 생성: urea-srs1${NC}"

# 기존 환경이 있다면 제거
if conda env list | grep -q "urea-srs1"; then
    echo "기존 urea-srs1 환경 제거 중..."
    conda env remove -n urea-srs1 -y
fi

# environment.yml로 환경 생성
if [ -f "environment.yml" ]; then
    echo "environment.yml로 가상환경 생성 중..."
    conda env create -f environment.yml
else
    echo "environment.yml 파일이 없습니다. 수동으로 환경 생성..."
    conda create -n urea-srs1 python=3.11 -y
    conda activate urea-srs1
    pip install -r requirements-ec2.txt
fi

# 5. 환경 활성화 및 패키지 확인
echo -e "${YELLOW}🔍 설치된 패키지 확인${NC}"
source $HOME/miniconda3/bin/activate urea-srs1

echo "Python 버전:"
python --version

echo "주요 패키지 버전:"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import lightgbm; print(f'lightgbm: {lightgbm.__version__}')"
python -c "import mlflow; print(f'mlflow: {mlflow.__version__}')"

# 6. 환경 활성화 스크립트 생성
echo -e "${YELLOW}📝 환경 활성화 스크립트 생성${NC}"
cat > activate_urea_env.sh << EOF
#!/bin/bash
# SRS1 요소수 제어 모델 환경 활성화
source \$HOME/miniconda3/bin/activate urea-srs1
echo "🐍 urea-srs1 환경이 활성화되었습니다!"
echo "Python 버전: \$(python --version)"
EOF

chmod +x activate_urea_env.sh

# 7. 완료 메시지
echo ""
echo -e "${GREEN}🎉 EC2 환경 설정 완료!${NC}"
echo "=================================================="
echo -e "${GREEN}📋 설정 결과:${NC}"
echo "  - 가상환경: urea-srs1 (Python 3.11)"
echo "  - 패키지: SRS1-sooeun과 동일한 버전"
echo "  - InfluxDB: 연결 라이브러리 설치됨"
echo ""
echo -e "${GREEN}🚀 다음 단계:${NC}"
echo "  1. 환경 활성화: source activate_urea_env.sh"
echo "  2. InfluxDB 연결 테스트: python test_scripts/test_influxdb_connection.py"
echo "  3. 모델 추론 테스트: python test_scripts/test_model_inference.py"
echo "  4. 통합 파이프라인 테스트: python test_scripts/test_realtime_pipeline.py"
echo ""
echo -e "${GREEN}✅ 스크립트 실행 완료!${NC}"
