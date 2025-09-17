#!/usr/bin/env python3
"""
CSV 파일을 Base64로 인코딩하여 텍스트로 전송할 수 있도록 하는 스크립트
"""

import base64
import sys
from pathlib import Path


def encode_csv_to_base64(csv_file):
    """CSV 파일을 Base64로 인코딩"""

    if not Path(csv_file).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {csv_file}")
        return

    try:
        file_path = Path(csv_file)
        file_size_kb = file_path.stat().st_size / 1024

        print(f"📊 파일: {csv_file}")
        print(f"📊 크기: {file_size_kb:.1f} KB")

        if file_size_kb > 1000:  # 1MB 제한
            print(
                f"⚠️ 파일이 {file_size_kb:.1f}KB로 너무 큽니다. 1MB 이하 파일만 지원합니다."
            )
            return

        # 파일을 Base64로 인코딩
        with open(csv_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        print("\n" + "=" * 80)
        print("📋 Base64 인코딩된 데이터 (아래 전체를 복사하세요):")
        print("=" * 80)

        # 80자씩 줄바꿈하여 출력
        for i in range(0, len(encoded), 80):
            print(encoded[i : i + 80])

        print("=" * 80)
        print("💡 로컬에서 디코딩 방법:")
        print(f"   1. 위 내용을 {file_path.stem}_encoded.txt로 저장")
        print(f"   2. Python에서 디코딩:")
        print(f"      import base64")
        print(f"      with open('{file_path.stem}_encoded.txt', 'r') as f:")
        print(f"          encoded = f.read().replace('\\n', '')")
        print(f"      with open('{csv_file}', 'wb') as f:")
        print(f"          f.write(base64.b64decode(encoded))")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python encode_csv.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    encode_csv_to_base64(csv_file)
