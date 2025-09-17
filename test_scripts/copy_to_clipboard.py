#!/usr/bin/env python3
"""
작은 CSV 파일을 터미널에 출력하여 복사할 수 있도록 하는 스크립트
"""

import pandas as pd
import sys
from pathlib import Path


def print_csv_for_copy(csv_file, max_rows=100):
    """CSV 파일을 터미널에 출력"""

    if not Path(csv_file).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {csv_file}")
        return

    try:
        df = pd.read_csv(csv_file)

        print(f"📊 파일: {csv_file}")
        print(f"📊 크기: {len(df)} 행, {len(df.columns)} 컬럼")

        if len(df) > max_rows:
            print(
                f"⚠️ 파일이 {len(df)}행으로 너무 큽니다. 처음 {max_rows}행만 출력합니다."
            )
            df = df.head(max_rows)

        print("\n" + "=" * 80)
        print("📋 아래 내용을 복사하여 로컬에서 CSV 파일로 저장하세요:")
        print("=" * 80)

        # CSV 형태로 출력
        print(df.to_csv(index=False))

        print("=" * 80)
        print(
            "💡 사용법: 위 내용을 복사 → 로컬에서 텍스트 파일로 저장 → .csv 확장자로 변경"
        )
        print("=" * 80)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python copy_to_clipboard.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    print_csv_for_copy(csv_file)
