from datetime import datetime, timedelta
import subprocess
import time  #  추가

# 기본 설정
base_start = datetime(2025, 10, 1, 0, 0, 0)
base_end   = datetime(2025, 10, 15, 0, 0, 0)
interval   = timedelta(hours=12)

i = 0
current = base_start

while current < base_end:
    start_time = current
    end_time = current + interval
    if end_time > base_end:
        end_time = base_end

    i += 1
    cmd = [
        "python3", "detect_model_drift.py",
        "--start-time", start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "--end-time", end_time.strftime("%Y-%m-%d %H:%M:%S"),
    ]

    print(f"\n  ({i}) 실행: {cmd}")
    subprocess.run(cmd, check=True)

    # 호출 간 1초 대기 (서버 부하 방지)
    time.sleep(1)

    current += interval

