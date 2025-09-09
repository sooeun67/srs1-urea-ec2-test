# ======================
# logger.py
# ======================
"""
공용 로거 유틸리티 모듈
- 콘솔 로깅 지원
- 노트북 환경 중복 핸들러 방지
| 임계값          | 표시되는 호출                                                  |
| -------------- | ------------------------------------------------------------ |
| `DEBUG`(10)    | `debug`, `info`, `warning`, `error`, `exception`, `critical` |
| `INFO`(20)     | `info`, `warning`, `error`, `exception`, `critical`          |
| `WARNING`(30)  | `warning`, `error`, `exception`, `critical`                  |
| `ERROR`(40)    | `error`, `exception`, `critical`                             |
| `CRITICAL`(50) | `critical` 만 표시 *(주의: `exception`은 `ERROR` 레벨이라 숨김)* |
"""

import logging
import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class LoggerConfig:
    name: str = "Script"
    level: int = logging.INFO
    fmt: str = "[%(asctime)s] [%(levelname)-5s] [%(name)-15s] %(message)s"
    # "%(asctime)s | %(levelname)-5s | %(name)-12s | %(funcName)-16s:%(lineno)-4d | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    propagate: bool = False
    refresh_handlers: bool = False
    use_stdout: bool = True


def get_logger(cfg: LoggerConfig) -> logging.Logger:
    """
    하이퍼파라미터 기반 로거 생성 함수
    """
    logger = logging.getLogger(cfg.name)

    # 기존 핸들러 제거
    if cfg.refresh_handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    # 이미 설정된 경우는 레벨만 조정하고 반환
    if logger.handlers and not cfg.refresh_handlers:
        logger.setLevel(cfg.level)
        logger.propagate = cfg.propagate
        return logger

    logger.setLevel(cfg.level)
    logger.propagate = cfg.propagate

    formatter = logging.Formatter(fmt=cfg.fmt, datefmt=cfg.datefmt)

    stream = sys.stdout if cfg.use_stdout else sys.stderr
    sh = logging.StreamHandler(stream)
    sh.setLevel(cfg.level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def log_series_stats(
    logger: logging.Logger,
    name: str,
    arr: np.ndarray,
    *,
    sample_stride: int = 1,
) -> None:
    """
    Lightweight numeric-array stats logger for training/EDA (DEBUG-only).

    왜 utils에 두나요?
    - Reuse: LGBM/GP 등 여러 모델/모듈에서 **공통으로 재사용**.
    - Single source of truth: 통계 로깅 포맷/정책을 **한 곳에서 관리**(중복 구현 제거).
    - Decoupling: 모델 코드와 분리해 **순환 의존성 방지**(model -> utils 단방향).
    - Testability: 모델과 분리되어 **단위 테스트**가 쉬움(입력/출력이 명확).
    - Performance control: **DEBUG 레벨일 때만 계산**하며, `sample_stride`로 비용 조절.

    Parameters
    ----------
    logger : logging.Logger
        사용 중인 로거 인스턴스. DEBUG가 아니면 즉시 반환(연산 스킵).
    name : str
        로그에 표기할 시리즈/컬럼 이름(예: "X[icf_tms_nox_a]").
    arr : np.ndarray
        1D 숫자 배열(Series.to_numpy 등). NaN 포함 가능.
    sample_stride : int, optional (default=1)
        큰 배열일 때 성능 비용을 낮추기 위한 **다운샘플 간격**.
        예) 10이면 10개 중 1개만 사용해 통계 계산(로그 목적엔 충분).

    Notes
    -----
    - 성능 안전장치: `logger.isEnabledFor(logging.DEBUG)`가 False면 **계산 자체를 하지 않음**.
    - 모든 값이 NaN이거나 길이가 0인 경우도 안전하게 처리.
    """
    # DEBUG 모드가 아니면 비용 발생 없이 즉시 반환
    if not logger.isEnabledFor(logging.DEBUG):
        return

    try:
        a = np.asarray(arr, dtype=float)
        n = a.size
        if n == 0:
            logger.debug(f"{name}: count=0 (empty)")
            return

        # 선택적 다운샘플링으로 대용량 시 비용 절감
        if sample_stride > 1 and n > sample_stride:
            a = a[::sample_stride]
            n = a.size

        nan_cnt = int(np.isnan(a).sum())
        if nan_cnt == n:
            logger.debug(f"{name}: count={n}, nan={nan_cnt} (all-NaN)")
            return

        v = a[~np.isnan(a)]
        # 한눈에 상태 파악 가능한 기본 통계(사분위 포함)
        logger.debug(
            f"{name}: count={n}, nan={nan_cnt}, "
            f"mean={np.mean(v):.6g}, std={np.std(v):.6g}, "
            f"min={np.min(v):.6g}, p25={np.percentile(v,25):.6g}, "
            f"median={np.median(v):.6g}, p75={np.percentile(v,75):.6g}, "
            f"max={np.max(v):.6g}"
        )
    except Exception as e:
        # 로깅 실패가 학습/추론을 막지 않도록 방어
        logger.debug(f"{name}: stats logging skipped ({e})")
