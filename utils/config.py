"""
SKEP Urea Control System - Configuration Loader
플랜트별 설정 파일을 동적으로 로드하는 역할만 담당
"""

import os
from dataclasses import dataclass
from importlib import import_module

import boto3

# 프로덕션 환경: 환경변수에서 플랜트 코드 로드
plant_code = os.environ.get("PLANT_CD")
print("Current plant code is:", plant_code)

# 플랜트 코드 매핑
if plant_code == "SRGG":
    plant_code = "SEGG"
elif plant_code == "SRS1":
    plant_code = "SRS1"

assert plant_code, f"plant_code is {plant_code}"

# AWS SSM 클라이언트 초기화
ssm = boto3.client("ssm")

# 플랜트별 설정 파일 동적 로드
try:
    plant_config = import_module(f"lambda_config.{plant_code}_CONFIG")
except ImportError as e:
    raise ImportError(f"Configuration file for plant {plant_code} not found: {e}")

# 플랜트별 설정 클래스들을 직접 노출
InferenceConfig = plant_config.InferenceConfig
ProcessConfig = plant_config.ProcessConfig
ModelConfig = plant_config.ModelConfig
KDSConfig = plant_config.KDSConfig

# InfluxDB 설정을 위한 헬퍼 함수
def get_influx_parameter(name: str) -> str:
    """SSM Parameter Store에서 InfluxDB 연결 정보 로드"""
    # 플랜트별 SSM 경로
    plant_code_v2 = plant_code
    if plant_code == "SEGG":
        plant_code_v2 = "SRGG"
    elif plant_code == "SBSH":
        plant_code_v2 = "SRS1"
    
    param_name = f"/zero4-wte/{plant_code_v2}/database/influx/{name}"
    return ssm.get_parameter(Name=param_name, WithDecryption=True)["Parameter"]["Value"]


# InfluxDB 설정 클래스
@dataclass
class InfluxConfig:
    """InfluxDB 연결 설정"""
    database_name: str = plant_code
    source_table_name: str = plant_code
    write_table_name: str = plant_code
    
    # SSM에서 연결 정보 로드
    host: str = get_influx_parameter("host")
    username: str = get_influx_parameter("id")
    password: str = get_influx_parameter("pw")
    port: str = get_influx_parameter("port")
    db_timeout: int = 30


# Urea 최적화 설정을 위한 헬퍼 함수
def get_urea_config(event_payload: dict | None = None):
    """플랜트별 Urea 설정 로드 (환경변수 및 이벤트 오버라이드 지원)"""
    # 플랜트별 UreaConfig 클래스가 있으면 사용, 없으면 기본값
    cfg_cls = getattr(plant_config, "UreaConfig", None)
    
    if cfg_cls is None:
        # 기본 Urea 설정
        @dataclass
        class DefaultUreaConfig:
            target_nox: float = 10.0
            p_feasible: float = 0.90
            min_hz: float = 38.0
            max_hz: float = 54.0
            round_to_int: bool = True
            
            def apply_event_overrides(self, payload: dict) -> "DefaultUreaConfig":
                if not isinstance(payload, dict):
                    return self
                for key, alias in [
                    ("target_nox", ["target_nox", "TARGET_NOX"]),
                    ("p_feasible", ["p_feasible", "P_FEASIBLE"]),
                    ("min_hz", ["min_hz", "MIN_HZ", "minimum_hz", "minimum"]),
                    ("max_hz", ["max_hz", "MAX_HZ", "maximum_hz", "maximum"]),
                    ("round_to_int", ["round_to_int", "ROUND_TO_INT"]),
                ]:
                    for name in alias:
                        if name in payload and payload[name] is not None:
                            setattr(self, key, payload[name] if key != "round_to_int" else bool(payload[name]))
                            break
                return self
        
        cfg_cls = DefaultUreaConfig
    
    cfg = cfg_cls() if callable(cfg_cls) else cfg_cls()
    
    # 환경변수 오버라이드
    try:
        env_map = {
            "target_nox": os.getenv("UREA_TARGET_NOX"),
            "p_feasible": os.getenv("UREA_P_FEASIBLE"),
            "min_hz": os.getenv("UREA_MIN_HZ"),
            "max_hz": os.getenv("UREA_MAX_HZ"),
            "round_to_int": os.getenv("UREA_ROUND_TO_INT"),
        }
        
        for k, v in env_map.items():
            if v is None:
                continue
            if k in ("target_nox", "p_feasible", "min_hz", "max_hz"):
                try:
                    setattr(cfg, k, float(v))
                except Exception:
                    pass
            elif k == "round_to_int":
                setattr(cfg, k, str(v).lower() in ("1", "true", "yes", "on"))
    except Exception:
        pass
    
    # 이벤트 페이로드 오버라이드
    if event_payload:
        cfg.apply_event_overrides(event_payload)
    
    return cfg
