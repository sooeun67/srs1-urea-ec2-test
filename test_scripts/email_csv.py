#!/usr/bin/env python3
"""
CSV 파일을 이메일로 전송하는 스크립트 (SMTP 접근 가능한 경우)
"""

import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from pathlib import Path
import os


def send_csv_email(
    csv_file, to_email, from_email=None, smtp_server="smtp.gmail.com", smtp_port=587
):
    """CSV 파일을 이메일로 전송"""

    if not Path(csv_file).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {csv_file}")
        return False

    if not from_email:
        from_email = input("발신자 이메일: ")

    password = input("이메일 비밀번호 (또는 앱 비밀번호): ")

    try:
        file_path = Path(csv_file)
        file_size_kb = file_path.stat().st_size / 1024

        if file_size_kb > 25000:  # 25MB 제한
            print(
                f"⚠️ 파일이 {file_size_kb:.1f}KB로 너무 큽니다. 25MB 이하 파일만 지원합니다."
            )
            return False

        # 이메일 메시지 생성
        msg = MIMEMultipart()
        msg["From"] = from_email
        msg["To"] = to_email
        msg["Subject"] = f"InfluxDB 데이터 내보내기: {file_path.name}"

        # 본문
        body = f"""
        InfluxDB에서 내보낸 데이터 파일입니다.
        
        파일명: {file_path.name}
        파일 크기: {file_size_kb:.1f} KB
        생성 시간: {file_path.stat().st_mtime}
        """

        msg.attach(MIMEText(body, "plain"))

        # 파일 첨부
        with open(csv_file, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {file_path.name}",
        )

        msg.attach(part)

        # SMTP 서버 연결 및 전송
        print(f"📧 이메일 전송 중... ({smtp_server}:{smtp_port})")

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()

        print(f"✅ 이메일 전송 완료: {to_email}")
        return True

    except Exception as e:
        print(f"❌ 이메일 전송 실패: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python email_csv.py <csv_file> <to_email> [from_email]")
        sys.exit(1)

    csv_file = sys.argv[1]
    to_email = sys.argv[2]
    from_email = sys.argv[3] if len(sys.argv) > 3 else None

    send_csv_email(csv_file, to_email, from_email)
