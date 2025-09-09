from __future__ import annotations

# === Standard library ===
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pprint import pformat
from typing import Optional, Sequence, Tuple

# === Third-party ===
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# === Local application ===
from config.column_config import ColumnConfig
from config.model_config import LGBMModelConfig
from config.preprocessing_config import TrainPreprocessingConfig
from utils.logger import get_logger, log_series_stats  # LoggerConfig는 model_config가 들고 있음




# -----------------------------
# 준비
# -----------------------------

# 안전한 지연 생성자
def _default_column_config():
    return ColumnConfig()

def _default_model_config():
    return LGBMModelConfig()




# -----------------------------
# LightGBM 래퍼 (Booster 지원)
# -----------------------------
@dataclass
class LGBMNOxModel:
    """
    NOx 예측을 위한 LightGBM 래퍼
    - column_config: ColumnConfig (또는 동형 객체)
    - model_config : LGBMModelConfig (to_lgbm_params(), model_path/meta_path/native_model_path 보유 권장)
    - 기능: fit / predict / predict_df / save / load / get_model_info / evaluate_validation
    - Booster 네이티브 모델(*.txt) 저장/로드 지원 (콜드 스타트 최적화)
    """

    column_config: ColumnConfig = field(default_factory=_default_column_config)
    model_config: LGBMModelConfig = field(default_factory=_default_model_config)

    # 런타임
    model: Optional[lgb.LGBMRegressor] = field(default=None, repr=False)
    _booster: Optional[lgb.Booster] = field(default=None, repr=False)  # ✅ Booster 핸들
    # 클래스 필드에 추가
    _train_df_: Optional[pd.DataFrame] = field(default=None, repr=False)
    _n_train_: Optional[int] = None
    _trained_at_: Optional[str] = None  # 학습 완료 시각(KST), "YYYY-MM-DD HH:MM:SS"

    # 로거 인스턴스(설정은 model_config.logger_cfg에서 주입)
    logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = get_logger(self.model_config.logger_cfg)

    # ----------------------
    # Helper: native model path 결정
    # ----------------------
    def _resolve_native_path(self, model_path: Optional[str] = None) -> str:
        """
        model_config.native_model_path가 있으면 우선 사용.
        없으면 model_path(또는 config.model_path)와 같은 폴더/이름으로 확장자만 .txt.
        """

        native_path = getattr(self.model_config, "native_model_path", None)
        if native_path:
            return native_path
        base_model_path = model_path or getattr(self.model_config, "model_path", "./artifacts/lgbm_model.joblib")
        root, _ = os.path.splitext(base_model_path)
        return f"{root}.txt"

    # ----------------------
    # 학습
    # ----------------------
    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str,
        sample_weight_col: Optional[str] = None,
    ) -> "LGBMNOxModel":
        """
        LightGBM 학습.
        - df: 학습 데이터프레임 (feature + target 포함)
        - sample_weight_col: 가중치 컬럼명(없으면 None)

        결측 처리 정책:
          • 타깃(y)만 NaN 드롭 (LightGBM 학습 요건)  ※ 아래에서 명시적으로 드롭
          • 피처(X) NaN은 드롭하지 않음 → LightGBM 내부 NaN 처리 사용
        """

        # === 환경 설정 (config, column 등) ===

        self.logger.info("LGBM Model 학습 시작")
        exec_file = os.path.basename(globals().get("__file__", "interactive"))
        self.logger.debug(f"실행 파일: {exec_file}")

        cc, mc = self.column_config, self.model_config

        feature_cols = getattr(mc, "lgbm_feature_columns_all", None) or list(cc.lgbm_feature_columns)
        missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
        if missing_cols:
            raise KeyError(f"입력 DF에 필요한 컬럼 누락: {missing_cols}")

        # === 데이터 준비 ===
        d = df.copy()

        # X 전처리: 숫자화 + inf -> NaN
        X = d[feature_cols].apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)

        # y 전처리: 숫자화 후 NaN 드롭
        y = pd.to_numeric(d[target_col], errors="coerce")

        # sample_weight 정합
        w = None
        if sample_weight_col is not None:
            if sample_weight_col not in d.columns:
                raise KeyError(f"가중치 컬럼 '{sample_weight_col}' 이(가) 없습니다.")
            w = pd.to_numeric(d[sample_weight_col], errors="coerce")

        # y NaN 마스크로 동시 필터링
        mask_y = ~np.isnan(y)
        X = X.loc[mask_y]
        y = y.loc[mask_y]
        if w is not None:
            w = w.loc[mask_y]

        # 스냅샷 저장(참고용)
        req_cols = list(dict.fromkeys(list(cc.lgbm_feature_columns) + [target_col] + ([sample_weight_col] if sample_weight_col else [])))

        self._train_df_ = d.loc[mask_y, req_cols].copy()
        self.logger.debug(f"학습 데이터 스냅샷 저장: shape={self._train_df_.shape}")
        
        # 간단 통계 로그 (DEBUG에서만 계산)
        self.logger.debug(f"X.shape={X.shape}, y.shape={y.shape}")
        if self.logger.isEnabledFor(logging.DEBUG):
            for i, col in enumerate(feature_cols):  # ← 실제 학습 피처 순서 기준
                colv = X.iloc[:, i].to_numpy(dtype=float, copy=False)
                log_series_stats(self.logger, f"X[{col}]", colv, sample_stride=1)  # 대용량이면 5~10로

            # target 통계 (target_col 사용 권장)
            log_series_stats(self.logger, f"y[{target_col}]", y.to_numpy(dtype=float, copy=False))

        # === fit 핵심 ===

        # 모델/파라미터
        params = mc.to_lgbm_params() if hasattr(mc, "to_lgbm_params") else {}
        self.model = lgb.LGBMRegressor(**params)
        self.logger.info("LightGBM 파라미터:")
        try:
            self.logger.info(json.dumps(params, ensure_ascii=False, indent=2))
        except Exception:
            self.logger.info(str(params))

        # 학습
        self.model.fit(X, y, sample_weight=w)
        self._n_train_ = int(len(X))
        self._booster = getattr(self.model, "booster_", None)

        # 학습 시각(KST)
        _kst = timezone(timedelta(hours=9))
        self._trained_at_ = datetime.now(_kst).strftime("%Y-%m-%d %H:%M:%S")

        self.logger.info("LGBM Model 학습 완료")
        self.logger.debug(f"학습 완료 시각(KST): {self._trained_at_}")
        return self


    # ----------------------
    # 예측 (ndarray 입력)
    # ----------------------

    # (LGBM) num_iteration 동기화
    def _resolve_num_iteration(self, num_iteration: Optional[int]) -> Optional[int]:
        if num_iteration is not None:
            return num_iteration
        if self.model is not None and getattr(self.model, "best_iteration_", None):
            return int(self.model.best_iteration_)
        if self._booster is not None and getattr(self._booster, "best_iteration", None):
            return int(self._booster.best_iteration)
        # 전체 트리
        return -1 if self._booster is not None else None

    # predict 내부
    def predict(
        self,
        X: np.ndarray,
        *,
        allow_nan: bool = True,
        num_iteration: Optional[int] = None,  # 두 경로 일치용
        lower_bound: Optional[float] = 0,
        upper_bound: Optional[float] = None
    ) -> np.ndarray:
        """
        학습된 모델로 예측 수행.
        - X: shape (n_samples, n_features)
        - allow_nan=False: '행 단위'로 NaN 검사 → NaN 있는 행만 NaN 출력, 나머지는 예측
        - num_iteration: sklearn/booster 경로 동일화를 위해 명시 지정 가능
        - lower_bound / upper_bound: 예측값 클리핑 범위 (None이면 미적용)
        """

        # === 환경 설정  ===

        self.logger.info("LGBM Model 예측 시작")
        exec_file = os.path.basename(globals().get("__file__", "interactive"))
        self.logger.debug(f"실행 파일: {exec_file}")

        if (self.model is None) and (self._booster is None):
            msg = "모델이 학습/로드되지 않았습니다. fit() 또는 load()를 먼저 호출하세요."
            self.logger.error(msg)
            raise ValueError(msg)

        if not isinstance(X, np.ndarray):
            msg = f"입력 타입 오류: numpy.ndarray 필요, 전달형={type(X)}"
            self.logger.error(msg)
            raise TypeError(msg)

        if X.ndim != 2:
            msg = f"입력 형상 오류: (n_samples, n_features) 필요, 전달형={X.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

        # expected feature size 확인 (가능하면)
        expected_features = None
        if self.model is not None:
            expected_features = int(getattr(self.model, "n_features_in_", X.shape[1]))
        elif self._booster is not None:
            try:
                expected_features = int(self._booster.num_feature())
            except Exception:
                expected_features = None

        if (expected_features is not None) and (X.shape[1] != expected_features):
            msg = f"입력 형상 오류: expected (n_samples, {expected_features}), got {X.shape}"
            self.logger.error(msg)
            raise ValueError(msg)

        # inf → NaN 치환
        X = np.asarray(X, dtype=float)
        X = np.where(np.isfinite(X), X, np.nan)

        num_iteration = self._resolve_num_iteration(num_iteration)

        # === predict: 결측치 허용 여부에 따라 if-else로 분기 ===

        if allow_nan:
            # 전체를 그대로 예측 (LightGBM 쪽 NaN 허용)
            if self._booster is not None:
                y_hat = self._booster.predict(X, num_iteration=num_iteration)
                self.logger.info("LGBM Model 예측 완료(booster, allow_nan=True)")
            else:
                y_hat = self.model.predict(X, num_iteration=num_iteration)
                self.logger.info("LGBM Model 예측 완료(sklearn, allow_nan=True)")
        else:
            # ✅ 행 단위 NaN 처리: NaN 있는 행만 NaN으로, 나머지는 예측
            valid_mask = ~np.isnan(X).any(axis=1)
            y_hat = np.full(X.shape[0], np.nan, dtype=float)

            if valid_mask.any():
                Xv = X[valid_mask]
                if self._booster is not None:
                    y_valid = self._booster.predict(Xv, num_iteration=num_iteration)
                    y_hat[valid_mask] = y_valid
                    self.logger.info("LGBM Model 예측 완료(booster, row-wise NaN handling)")
                else:
                    y_valid = self.model.predict(Xv, num_iteration=num_iteration)
                    y_hat[valid_mask] = y_valid
                    self.logger.info("LGBM Model 예측 완료(sklearn, row-wise NaN handling)")
            else:
                self.logger.warning("모든 행에 NaN이 포함되어 있어 전부 NaN을 반환합니다.")

        # === postprocess: bounds 적용 ===
        if lower_bound is not None:
            y_hat = np.maximum(y_hat, lower_bound)
        if upper_bound is not None:
            y_hat = np.minimum(y_hat, upper_bound)

        return y_hat


    # ----------------------
    # 예측 (DataFrame 입력)
    # ----------------------
    def predict_df(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: Optional[Sequence[str]] = None,
        allow_nan: bool = False,
        num_iteration: Optional[int] = None,
        output_name: str = "y_pred",
    ) -> pd.Series:
        """
        DataFrame에서 feature 컬럼을 추출해 예측합니다.

        사용 목적:
        - `predict`: ndarray 입력 전용 (고성능/실시간 inference에 적합)
        - `predict_df`: DataFrame 입력 전용 (오프라인 검증/EDA, 
          feature 컬럼 매핑·결측 처리·안전성 체크에 유리)

        동작:
        - feature_cols가 None이면 self.column_config.lgbm_feature_columns 사용
        - 숫자 변환(pd.to_numeric, errors='coerce') 후 inf→NaN 치환
        - allow_nan=False면 행 단위 NaN 검사 후 NaN 있는 행만 NaN 반환(내부 predict와 동일 정책)
        - 반환: 예측값 Series(index=df.index, name=output_name)

        권장 사용 시나리오:
        - 실시간/온라인 추론 → `predict`
            (입력이 이미 ndarray로 준비되어 있고, 속도가 중요한 경우)
        - 오프라인 검증/모델 평가 → `predict_df`
            (DataFrame 기반으로 feature를 안전하게 선택/검증하고,
            NaN·inf 처리 로직을 포함해 안정적으로 예측할 수 있음)
        """

        # 1) 피처 목록 확정
        feats = list(feature_cols) if feature_cols is not None else list(self.column_config.lgbm_feature_columns)

        # 2) 누락 컬럼 검사
        missing = [c for c in feats if c not in df.columns]
        if missing:
            msg = f"예측에 필요한 컬럼 누락: {missing}"
            self.logger.error(msg)
            raise KeyError(msg)

        # 3) 숫자 변환 및 ndarray 생성(열 순서 고정)
        X = (
            df.loc[:, feats]
              .apply(pd.to_numeric, errors="coerce")
              .to_numpy(dtype=float, copy=False)
        )

        # 4) inf → NaN 치환
        np.place(X, ~np.isfinite(X), np.nan)

        # === 간단한 로깅 ===
        n_total = X.shape[0]
        n_nan_rows = np.isnan(X).any(axis=1).sum()
        ratio = (n_nan_rows / n_total) if n_total else 0.0
        self.logger.info(
            f"predict_df 입력: samples={n_total}, features={X.shape[1]}, "
            f"NaN rows={n_nan_rows} ({ratio:.1%})"
        )

        # 5) 예측
        y_hat = self.predict(X, allow_nan=allow_nan, num_iteration=num_iteration)

        # 6) 출력 로깅
        self.logger.info(
            f"predict_df 출력 완료: y_pred shape={y_hat.shape}, "
            f"nan_count={np.isnan(y_hat).sum()}"
        )

        # 7) Series로 반환(입력 DF index 유지)
        return pd.Series(y_hat, index=df.index, name=output_name)


    # ----------------------
    # 저장 (joblib + booster + meta)
    # ----------------------
    
    # 직렬화 헬퍼: numpy/datetime/집합형 등을 안전하게 문자열/리스트로 변환
    @staticmethod
    def _json_default(o):
        import numpy as np
        from datetime import datetime, date
        # numpy 스칼라
        if isinstance(o, np.generic):
            return o.item()
        # numpy 배열
        if isinstance(o, np.ndarray):
            return o.tolist()
        # datetime/date
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        # set/tuple 등 순회 가능한 집합형
        if isinstance(o, (set, tuple)):
            return list(o)
        # dataclass/객체 등: dict로 떨어지면 dict, 아니면 문자열
        if hasattr(o, "__dict__"):
            try:
                return vars(o)
            except Exception:
                pass
        return str(o)

    def save(self, model_path: Optional[str] = None, meta_path: Optional[str] = None) -> Tuple[str, str]:
        """
        모델 및 메타 저장.
        - model_path 없음: model_config.model_path
        - meta_path  없음: model_config.meta_path
        - 추가: Booster 네이티브 모델도 함께 저장 (*.txt)
        """
        self.logger.info("LGBM Model 저장 시작")

        if (self.model is None) and (self._booster is None):
            msg = "모델이 학습되지 않았습니다."
            self.logger.error(msg)
            raise ValueError(msg)

        model_path = model_path or getattr(self.model_config, "model_path", "./artifacts/lgbm_model.joblib")
        meta_path  = meta_path  or getattr(self.model_config, "meta_path",  "./artifacts/lgbm_meta.json")
        native_path = self._resolve_native_path(model_path)

        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(meta_path)  or ".", exist_ok=True)

        # 1) 모델 저장 (sklearn 래퍼가 있으면 저장)
        if self.model is not None:
            joblib.dump(self.model, model_path)
            self.logger.info(f"모델 저장 완료: {model_path}")
        else:
            self.logger.info("sklearn 모델 객체가 없어 joblib 저장을 생략합니다(booster만 사용).")

        # 2) Booster 네이티브 저장 (가능하면 항상 저장)
        try:
            booster: Optional[lgb.Booster] = self._booster or getattr(self.model, "booster_", None)
            if booster is not None:
                booster.save_model(native_path)
                self.logger.info(f"네이티브 모델 저장 완료: {native_path}")
        except Exception:
            self.logger.warning("네이티브 모델 저장 실패(무시).", exc_info=True)

        # 3) 메타 저장
        feature_cols_for_meta = getattr(self.model_config, "lgbm_feature_columns_all", None) \
                                or list(self.column_config.lgbm_feature_columns)

        meta = {
            "status": "trained",
            "trained_at": self._trained_at_,
            "n_train": self._n_train_,
            "n_features": (
                int(getattr(self.model, "n_features_in_", 0))
                if self.model is not None else
                int(self._booster.num_feature()) if self._booster is not None else 0
            ),
            "feature_columns": list(feature_cols_for_meta),
            "target_column": self.column_config.target_column,
            "lgbm_params": (self.model.get_params() if self.model is not None else {}),
            "config": asdict(self.model_config) if hasattr(self.model_config, "__dataclass_fields__") else {},
            "native_model_path": native_path,
        }

        # with open(meta_path, "w", encoding="utf-8") as f:
        #     json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=self._json_default)

        self.logger.info(f"메타 저장 완료: {meta_path}")

        # (선택) 학습 스냅샷 CSV 저장
        try:
            if self._train_df_ is not None:
                base = os.path.splitext(os.path.basename(model_path))[0]
                csv_path = os.path.join(os.path.dirname(model_path) or ".", f"{base}_train.csv")
                self._train_df_.to_csv(csv_path, index=False, encoding="utf-8-sig")
                self.logger.info(f"학습 데이터 CSV 저장 완료: {csv_path} (rows={len(self._train_df_)})")
        except Exception:
            self.logger.warning("학습 데이터 CSV 저장 중 예외(무시).", exc_info=True)

        return model_path, meta_path


    # ----------------------
    # 로드 (네이티브 우선)
    # ----------------------
    def load(self, model_path: Optional[str] = None) -> "LGBMNOxModel":
        """
        모델 로드.
        - Booster 네이티브 모델이 존재하면 우선 사용(빠름)
        - 없으면 joblib sklearn 래퍼 로드
        """
        self.logger.info("LGBM Model 로드 시작")

        # 경로 확정(1회)
        resolved_model_path = model_path or getattr(self.model_config, "model_path", "./artifacts/lgbm_model.joblib")
        native_path = self._resolve_native_path(resolved_model_path)

        if os.path.exists(native_path):
            self._booster = lgb.Booster(model_file=native_path)
            self.model = None
            self.logger.info(f"네이티브 Booster 로드 완료: {native_path}")
        elif os.path.exists(resolved_model_path):
            self.model = joblib.load(resolved_model_path)
            self._booster = getattr(self.model, "booster_", None)
            self.logger.info(f"모델 로드 완료: {resolved_model_path}")
        else:
            msg = f"모델 파일을 찾을 수 없습니다: {resolved_model_path}"
            self.logger.error(msg)
            raise FileNotFoundError(msg)
        
        return self


    # ----------------------
    # 메타 정보
    # ----------------------
    def get_model_info(self) -> dict:
        """모델 메타 정보 조회."""
        self.logger.info("LGBM Model 정보 조회 시작")
        exec_file = os.path.basename(globals().get("__file__", "interactive"))
        self.logger.debug(f"실행 파일: {exec_file}")

        if (self.model is None) and (self._booster is None):
            self.logger.warning("모델 정보 요청: 학습 전 상태(not_trained)")
            return {"status": "not_trained"}

        n_features = (
            int(getattr(self.model, "n_features_in_", 0))
            if self.model is not None else
            int(self._booster.num_feature()) if self._booster is not None else 0
        )
        params = self.model.get_params() if self.model is not None else {}

        info = {
            "status": "trained",
            "backend": "booster" if (self._booster is not None and self.model is None) else "sklearn",
            "n_features": n_features,
            "n_train": self._n_train_,
            "random_state": getattr(self.model_config, "lgbm_random_state", None),
            "trained_at": self._trained_at_,
            "feature_columns": list(self.column_config.lgbm_feature_columns),
            "target_column": self.column_config.target_column,
            "objective": getattr(self.model_config, "lgbm_objective", None),
            "params": params,
            "has_booster": self._booster is not None,
        }

        self.logger.debug("모델 정보 상세:\n" + pformat(info, width=120))
        self.logger.info(
            f"model_info: status={info['status']}, backend={info['backend']}, "
            f"n_features={info['n_features']}, n_train={info['n_train']}"
        )
        self.logger.info("LGBM Model 정보 조회 완료")

        return info