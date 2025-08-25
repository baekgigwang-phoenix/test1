# 로봇 쪽 테스트 코드

import socket

HOST = "0.0.0.0"  # 모든 네트워크에서 대기
PORT = 5000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"[INFO] 로봇 서버 실행 중... (Port {PORT})")

try:
    while True:
        conn, addr = server_socket.accept()
        print(f"[INFO] 연결됨: {addr}")

        data = conn.recv(1024).decode().strip()
        print(f"[RECV] {data}")

        # 여기서 data 파싱 후 로봇 제어 코드 실행 가능
        conn.sendall(f"ACK: {data}".encode())
        conn.close()
except KeyboardInterrupt:
    print("\n[INFO] 서버 종료")
finally:
    server_socket.close()
