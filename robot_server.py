import socket
import cv2
import numpy as np
import time
from pymycobot.mycobot280 import MyCobot280

# ------------------------------
# 로봇 동작 함수 (PnP)
# ------------------------------
def pick_and_place(marker_id):
    print(f"[INFO] 마커 ID {marker_id} 작업 시작")

    # ----- 1. 로봇 연결 및 초기화 -----
    mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
    mc.thread_lock = True
    initial_angles = [-3.77, 109.07, -133.41, 21.44, 5.88, -44.91]
    start_angles = [-2.02, 10.98, -130.78, 125, 1.31, -46.31]
    speed = 30

    mc.send_angles(initial_angles, speed)
    mc.set_gripper_value(100, speed)
    time.sleep(1)

    # ----- 2. 카메라 초기화 -----
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ----- 3. ArUco 마커 설정 -----
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # ----- 4. 카메라 보정 데이터 로드 -----
    calib_data = np.load('/home/jetcobot/cam_calib/calib_intrinsic.npz')
    K = calib_data['mtx']
    dist = calib_data['dist']

    marker_length = 0.02
    objp = np.array([
        [-marker_length/2,  marker_length/2, 0],
        [ marker_length/2,  marker_length/2, 0],
        [ marker_length/2, -marker_length/2, 0],
        [-marker_length/2, -marker_length/2, 0]
    ], dtype=np.float32)

    # ----- 5. 대상 마커 탐지 -----
    found = False
    while not found:
        ret, frame = cap.read()
        if not ret:
            continue

        corners, ids, _ = detector.detectMarkers(frame)
        if ids is not None:
            ids = ids.flatten()
            for i, mid in enumerate(ids):
                if mid == marker_id:
                    imgp = corners[i][0].astype(np.float32)
                    success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
                    if not success:
                        continue

                    # mm 변환
                    x_c, y_c, z_c = [round(float(val), 2) for val in tvec.reshape(-1) * 1000]
                    z_c -= 55.37

                    coords_o = mc.get_coords()
                    if coords_o is None:
                        print("[ERROR] 로봇 좌표 수신 실패")
                        return

                    # 이동 경로 설정
                    t1 = coords_o.copy()
                    t1[0] += 10
                    t1[1] -= x_c
                    t1[2] -= y_c
                    t1[2] += 80
                    t1 = [round(val, 2) for val in t1]

                    mc.send_coords(t1, speed, 0)
                    time.sleep(2)

                    t2 = t1.copy()
                    t2[0] += 240
                    t2[2] += 20
                    t2 = [round(val, 2) for val in t2]

                    mc.send_coords(t2, speed, 0)
                    time.sleep(3)

                    mc.set_gripper_value(0, 50)
                    time.sleep(2)

                    mc.send_coords(coords_o, speed, 0)
                    time.sleep(2)

                    place_pos = [89.91, -9.84, -95.8, 37.0, 4.48, -48.6]
                    mc.send_angles(place_pos, 20, 0)
                    time.sleep(2)

                    mc.set_gripper_value(100, 50)
                    time.sleep(2)

                    mc.send_angles(start_angles, speed, 0)
                    time.sleep(2)

                    found = True
                    break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] 마커 ID {marker_id} 작업 완료")


# ------------------------------
# TCP 서버 실행
# ------------------------------
HOST = "0.0.0.0"  # 모든 인터페이스에서 수신
PORT = 5000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print(f"[INFO] 서버 시작: {HOST}:{PORT}")

client_socket, addr = server_socket.accept()
print(f"[INFO] 클라이언트 접속: {addr}")

while True:
    data = client_socket.recv(1024).decode().strip()
    if not data:
        break

    print(f"[RECV] {data}")
    client_socket.sendall(f"ACK: {data}".encode())

    if data.startswith("start"):
        _, marker_id = data.split(",")
        marker_id = int(marker_id)

        pick_and_place(marker_id)

        # 동작 완료 후 메시지 전송
        client_socket.sendall("적재 완료".encode())
