#!/usr/bin/env python3
"""
요소수 자동제어 모델 드리프트 감지용 성능 지표 계산
(1) LightGBM 예측성능(MAE): 
- 성능 지표: MAE
- 비교 대상: 1분 뒤 실제값
- 평가 대상: 현재 예측값
- (현재 예측값), (1분 뒤 실제값) 2개가 모두 있는 경우만 분석
(2) 요소수 절감률 계산
- 비교 대상: 60Hz 운영 시 유량 (상수값으로 고정, 별도 계산 과정 없음)
- 평가 대상: 평균 유량
"""

##############################
########## 환경 설정 ##########
##############################

import argparse

from datetime import datetime, timedelta
from influxdb import InfluxDBClient
import numpy as np
import pandas as pd



##############################
########## 하드코딩 ###########
##############################

baseline_urea_flow_rate = 63.73

# DB에서 read할 input column 목록
col_datetime = '_time_gateway'
col_inc_status = 'IncineratorStatus'
col_nox_eq_status = 'NOX_EQ_Status'
col_nox = 'ICF_TMS_NOX_A'
col_lgbm_db_pred_nox = 'SNR_NOX_PRED'
col_urea_pump = 'SNR_PMP_UW_S_1'
col_urea_flow = 'SNR_EQ_UW_F_1'


cols_select = [col_datetime, col_inc_status, col_nox_eq_status, col_nox, col_lgbm_db_pred_nox, col_urea_pump, col_urea_flow]

# DB에 update할 output column 목록
col_urea_saving_rate = 'SNR_UREA_SAVING_RATE'
col_snr_nox_mae = 'SNR_NOX_PRED_MAE'



##############################
########## 인자 파싱 ##########
##############################
print("")

parser = argparse.ArgumentParser(
    description="요소수 자동제어 모델 드리프트 감지용 성능 지표 계산 (함수 없이 단일 스크립트)"
)

# 조회 기간 설정
parser.add_argument(
    "--start-time",
    type=str,
    default=None,
    help="조회 시작시각 (UTC) 형식: 'YYYY-MM-DD HH:MM:SS' (예: '2025-10-13 00:00:00')",
)
parser.add_argument(
    "--end-time",
    type=str,
    default=None,
    help="조회 종료시각 (UTC) 형식: 'YYYY-MM-DD HH:MM:SS' (예: '2025-10-13 12:00:00'). 미지정 시 현재 UTC.",
)
parser.add_argument(
    "--query-hours",
    type=float,
    default=12.0,
    help=f"end_time 기준 과거 조회 시간(시간 단위). start_time 미지정 시 사용. 기본 12시간.",
)

# DB 관련 정보
# parser.add_argument("--influx-host", type=str, default="10.238.24.150") # 개발계
parser.add_argument("--influx-host", type=str, default="10.238.27.132") # 운영계
parser.add_argument("--influx-port", type=int, default=8086)
parser.add_argument("--influx-user", type=str, default="read_user")
parser.add_argument("--influx-pass", type=str, default="!Skepinfluxuser25")
parser.add_argument("--influx-db"  , type=str, default="SRS1")
parser.add_argument("--measurement", type=str, default="SRS1")

args = parser.parse_args()

# 기본값 딕셔너리 (argument 정의에서 설정된 default 값)
defaults = {a.dest: a.default for a in parser._actions if a.dest != "help"}

print("### Argument values ###")
for k, v in vars(args).items():
    default_v = defaults.get(k)
    if v == default_v:
        print(f"{k:15s}: {v}   (default)")
    else:
        print(f"{k:15s}: {v}   (input)")



################################
########## 시간 해석 ###########
################################

# end_time
if args.end_time:
    end_time = datetime.strptime(args.end_time, "%Y-%m-%d %H:%M:%S")
else:
    end_time = datetime.utcnow()

# query_hours
query_hours = float(args.query_hours)
if query_hours <= 0:
    raise ValueError("--query-hours must be positive.")
    
# start_time
if args.start_time:
    start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M:%S")
else:
    start_time = end_time - timedelta(hours=query_hours)

# 4) 유효성 체크
if start_time >= end_time:
    raise ValueError("start_time must be earlier than end_time.")

# 5) Influx 질의용 ISO8601 Z 포맷 문자열
start_time = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
end_time   = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")



#################################
########## Influx 조회 ##########
#################################
print("")

# InfluxDB 연결
client = InfluxDBClient(
    host=args.influx_host,
    port=args.influx_port,
    username=args.influx_user,
    password=args.influx_pass,
    database=args.influx_db
)
print(f" InfluxDB 연결: {args.influx_host}:{args.influx_port}/{args.influx_db}")

# SELECT 필드
select_fields = ", ".join([f'"{col}"' for col in cols_select])

query = (
    f'SELECT {select_fields} '
    f'FROM "{args.measurement}" '
    f"WHERE time >= '{start_time}' AND time < '{end_time}'"
)

print(f" 실행 쿼리:")
print(query)
print()

result = client.query(query)
df = pd.DataFrame(list(result.get_points()))

if df.empty:
    print(" 조회된 데이터가 없습니다.")

print(f" 조회 완료: {len(df)} 행, {len(df.columns)} 컬럼")

# 시간 컬럼 변환
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"])
    print(f" 데이터 시간 범위: {df['time'].min()} ~ {df['time'].max()}")

# 샘플 데이터 출력
print("샘플 데이터 출력 (head, tail)")
print(df.head())
print(df.tail())
    


#################################################
########## 비정상 가동 구간 데이터 제외 ###########
#################################################
print("")

required_cols = [col_inc_status, col_nox_eq_status]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f" 누락 컬럼: {missing} - 해당 필터를 건너뜁니다.")
else:
    # 정상 가동(IncineratorStatus != 0) 유지
    df = df.loc[df[col_inc_status] == 0]

    # NOx 장비 정상 상태만 유지 (== 0)
    df = df.loc[df[col_nox_eq_status] == 0]
    
# 같은 시각 중복 로우(초 단위) 평균으로 집계  정합 안정화
df = (
    df.set_index("time")
      .groupby(level=0)
      .mean(numeric_only=True)
      .sort_index()
      .reset_index()
)

print(f" 필터/정렬/중복처리 후: {len(df)} 행")



################################################
########## 성능 지표 (1) LightGBM MAE ###########
################################################
print("")

df = df.dropna(subset=['time'])
base_index = df.set_index("time")

missing_mae_cols = [col_lgbm_db_pred_nox, col_nox]
for c in missing_mae_cols:
    if c not in base_index.columns:
        raise ValueError(f"MAE 계산에 필요한 컬럼 누락: {c}")        

# 예측(t) & 실제(t+1분  인덱스를 -1분 시프트)
s_pred   = pd.to_numeric(base_index[col_lgbm_db_pred_nox], errors="coerce").rename("pred")
s_actual = pd.to_numeric(base_index[col_nox],               errors="coerce").rename("actual")

actual_tplus1 = s_actual.copy()
actual_tplus1.index = actual_tplus1.index - pd.Timedelta(minutes=1)

# 두 값이 "정확히 동시에 존재"하는 경우만 사용
aligned = pd.concat([s_pred, actual_tplus1.rename("actual_tplus1")], axis=1).dropna()

# 직접 MAE 계산 (sklearn 미사용)
if len(aligned) == 0:
    print("[METRIC] 정합된 표본이 없습니다. (동일 타임스탬프 t의 예측과 t+1의 실제가 동시에 없어요)")
else:
    abs_err = (aligned["pred"] - aligned["actual_tplus1"]).abs()
    mae = abs_err.mean()
    print(f"[METRIC] LGBM 1-min MAE (N={len(aligned)}): {mae:.4f}")



#####################################################
########## (참고) NOx 실제값 구간별 성능지표 ##########
#####################################################
print("")

if len(aligned) == 0:
    print("[BINNED] 정합된 표본이 없어 구간별 통계를 계산하지 않습니다.")
else:
    # aligned: index=time, columns=['pred','actual_tplus1']
    df_eval = aligned.reset_index().rename(columns={"index": "time"})
    df_eval = df_eval.rename(columns={"actual_tplus1": "target"})  # 실제값(t+1)을 target으로
    # 필요 시 시간 컬럼 이름을 문서용으로 맞추고 싶으면 아래 라인 사용 (옵션)
    # df_eval[col_datetime] = df_eval["time"]

    # NOx 구간(bins) 정의: 예) 0~120 ppm을 10ppm 간격으로
    # 범위를 데이터에 맞춰 유연하게 하려면 아래 두 줄로 자동 범위 산출 가능
    lo = max(0, np.floor(df_eval["target"].min() // 10) * 10)
    hi = np.ceil(df_eval["target"].max() / 10) * 10 + 10
    bins = np.arange(lo, hi + 1e-9, 10)  # 10ppm 간격

    df_eval["NOx_bin"] = pd.cut(df_eval["target"], bins=bins, right=False)

    # 예측오차 계산에 쓸 편의 변수
    err = df_eval["target"] - df_eval["pred"]
    abs_err = err.abs()

    # sMAPE 분모 0 안전처리
    denom = (df_eval["target"].abs() + df_eval["pred"].abs()).replace(0, np.nan)

    grouped = df_eval.groupby("NOx_bin", observed=False).apply(
        lambda g: pd.Series({
            "count": len(g),
            "ME": (g["target"] - g["pred"]).mean(),                                        # Mean Error
            "MAE": (g["target"] - g["pred"]).abs().mean(),                                 # Mean Abs Error
            "RMSE": np.sqrt(((g["target"] - g["pred"])**2).mean()),                        # Root MSE
            "sMAPE(%)": (2 * (g["target"] - g["pred"]).abs() / 
                         (g["target"].abs() + g["pred"].abs()).replace(0, np.nan)).mean() * 100,
            "pos_residual_count": (g["target"] - g["pred"] > 0).sum(),
            "neg_residual_count": (g["target"] - g["pred"] < 0).sum(),
            "pos_residual_ratio": (g["target"] - g["pred"] > 0).mean(),
            "neg_residual_ratio": (g["target"] - g["pred"] < 0).mean(),
        }),
        include_groups=False
    ).reset_index()

    print("\n[BINNED] Performance by NOx(target) bins:")
    # 보기 좋게 정렬
    try:
        grouped = grouped.sort_values(by="NOx_bin")
    except Exception:
        pass
    print(grouped.to_string(index=False))

    
    
#####################################################
########## 성능 지표 (2) 요소수 유량 절감률 ###########
#####################################################

print("")

# 평균 펌프 Hz 계산
if col_urea_pump not in df.columns:
    print(f"[PUMP] 컬럼 없음: {col_urea_pump}")
else:
    s_pump = pd.to_numeric(df[col_urea_pump], errors="coerce").dropna()
    if s_pump.empty:
        print("[PUMP] 펌프 주파수 데이터가 없습니다.")
    else:
        mean_hz = s_pump.mean()
        print(f"[PUMP] 평균 주파수 (Hz): {mean_hz:.2f} (N={len(s_pump)})")


# 평균 유량 계산
if col_urea_flow not in df.columns:
    print(f"[FLOW] 컬럼 없음: {col_urea_flow}")
else:
    s_flow = pd.to_numeric(df[col_urea_flow], errors="coerce").dropna()
    if s_flow.empty:
        print("[FLOW] 유량 데이터가 없습니다.")
    else:
        mean_urea = s_flow.mean()
        print(f"[FLOW] 평균 유량: {mean_urea:.4f} (N={len(s_flow)})")

        # 절감률 계산 (원하시는 경우에만)
        if isinstance(baseline_urea_flow_rate, (int, float)) and baseline_urea_flow_rate > 0:
            saving_rate = (baseline_urea_flow_rate - mean_urea) / baseline_urea_flow_rate
            print(f"[FLOW] 베이스라인 유량(60Hz 기준, {baseline_urea_flow_rate}) 대비 절감률: {saving_rate:.2%}")
        else:
            print("[FLOW] baseline_urea_flow_rate가 0이거나 유효하지 않아 절감률을 계산하지 않습니다.")



#############################################
########## 최종 요약 DataFrame (1행) #########
#############################################

# saving_rate / mae 가 계산되지 않았을 수도 있으니 안전하게 처리
urea_saving = None
if "saving_rate" in locals():
    try:
        urea_saving = float(saving_rate)
    except Exception:
        urea_saving = None

mae_value = None
if "mae" in locals():
    try:
        mae_value = float(mae)
    except Exception:
        mae_value = None

#  end_time 포맷 변환
end_time_str = pd.to_datetime(end_time).strftime("%Y-%m-%d %H:%M:%S")
summary_df = pd.DataFrame(
    [{
        col_datetime: end_time_str,              # 예: '2025-10-15 12:00:00':
        col_urea_saving_rate: urea_saving,       # float or None
        col_snr_nox_mae: mae_value,              # float or None
    }],
    columns=[col_datetime, col_urea_saving_rate, col_snr_nox_mae]
)

print("\n[RESULT] Summary one-row DataFrame:")
print(summary_df)


