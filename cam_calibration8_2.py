import cv2
import numpy as np
import time
from pymycobot.mycobot280 import MyCobot280
from scipy.spatial.transform import Rotation as R

import argparse
import imutils
import sys


# ----- 1. JetCobot 연결 및 초기 위치 설정 -----
mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
mc.thread_lock = True
print("[INFO] 로봇이 연결되었습니다.")

initial_angles = [-3.77, 109.07, -133.41, 21.44, 5.88, -44.91]
start_angles = [-2.02, 10.98, -130.78, 125, 1.31, -46.31]
# start_angles = [-4.21, 16.52, -136.93, 94.13, 5.36, -45.87]
# start_angles = [-2.02, 10.98, -130.78, 125, 1.31, -46.31]
speed = 30

print("[INFO] 초기 자세로 이동 중...")
mc.send_angles(initial_angles, speed)
mc.set_gripper_value(100, speed)
time.sleep(1)


# print("[INFO] 시작 좌표로 이동 중...")
# mc.send_angles(start_angles, speed, 0)
# time.sleep(1)

mc.set_gripper_value(100, 50)
time.sleep(1)

# ----- 2. 카메라 초기화 -----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ----- 3. ArUco 마커 탐지기 설정 -----
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ----- 4. 카메라 캘리브레이션 데이터 로드 -----
calib_data = np.load('/home/jetcobot/cam_calib/calib_intrinsic.npz')
K = calib_data['mtx']
dist = calib_data['dist']

marker_length = 0.02  # 마커 한 변 (m)
objp = np.array([
    [-marker_length/2,  marker_length/2, 0],
    [ marker_length/2,  marker_length/2, 0],
    [ marker_length/2, -marker_length/2, 0],
    [-marker_length/2, -marker_length/2, 0]
], dtype=np.float32)

# ----- 5. 마커 탐색 화면 출력 -----
print("[INFO] 카메라 영상 스트리밍 시작...")
last_print_time = 0
target_id = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 카메라 프레임 수신 실패")
        break

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        ids = ids.flatten()
        for marker_corner, marker_id in zip(corners, ids):
            pts = marker_corner.reshape((4, 2)).astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(frame, f"ID {marker_id}", (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if time.time() - last_print_time > 5:
            print(f"[INFO] 인식된 마커 ID들: {ids}")
            last_print_time = time.time()
    else:
        cv2.putText(frame, "no marker", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "Press 'q' to select target ID", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Marker Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----- 6. 추적할 마커 ID 입력 -----
cv2.destroyWindow("Marker Detection")
while True:
    try:
        target_id = int(input("[INPUT] 추적할 마커 ID를 입력하세요: "))
        break
    except ValueError:
        print("숫자로 입력해주세요.")

# ----- 7. 선택한 마커의 위치 추정 -----
ret, frame = cap.read()
cap.release()
cv2.destroyAllWindows()

if ret:
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        ids = ids.flatten()
        for i, marker_id in enumerate(ids):
            if marker_id != target_id:
                continue

            imgp = corners[i][0].astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
            if not success:
                print("[ERROR] solvePnP 실패")
                continue

            # t_m2c = tvec.reshape(3, 1)

            tvec = np.array(tvec)
            # x_c, y_c, z_c = tvec.reshape(-1) * 1000  # mm 단위
            x_c, y_c, z_c = [round(float(val), 2) for val in tvec.reshape(-1) * 1000]
            z_c -= 55.37

            coords = mc.get_coords()
            coords_o = mc.get_coords()
            if coords is None:
                print("[ERROR] 로봇 좌표 수신 실패")
                break

            print(coords)

            t1_coords = coords.copy()
            t1_coords[0] += 10
            t1_coords[1] -= x_c
            t1_coords[2] -= y_c
            t1_coords[2] += 80
            # t1_coords[2] += 20
            # t1_coords[0] += z_c
            # t1_coords[0] -= 20
            t1_coords = [round(val, 2) for val in t1_coords]
            # t1_coords = [float(val) for val in t1_coords]
            print(f"[INFO] t1으로 이동합니다: {t1_coords}")
            mc.send_coords(t1_coords, speed, 0)
            time.sleep(2)

            t2_coords = t1_coords.copy()
            t2_coords[0] += 240
            t2_coords[2] += 20
            # t2_coords[3] += 10
            t2_coords = [round(val, 2) for val in t2_coords]
            # t1_coords = [float(val) for val in t1_coords]
            print(f"[INFO] t2으로 이동합니다: {t2_coords}")
            mc.send_coords(t2_coords, speed, 0)
            time.sleep(3)

            # t2_coords = t1_coords.copy()
            # t2_coords[0] += 250
            # t2_coords[2] += 30
            # # t2_coords[3] += 10
            # t2_coords = [round(val, 2) for val in t2_coords]
            # # t1_coords = [float(val) for val in t1_coords]
            # print(f"[INFO] t2으로 이동합니다: {t2_coords}")
            # mc.send_coords(t2_coords, speed, 0)
            # time.sleep(2)

            print("[INFO] 물건 잡기")
            mc.set_gripper_value(0, 50)
            time.sleep(2)

            # t3_coords = t2_coords.copy()
            # t3_coords[0] -= z_c
            # # t2_coords[2] += 10
            # t3_coords = [round(val, 2) for val in t3_coords]
            # # t1_coords = [float(val) for val in t1_coords]
            # print(f"[INFO] t3으로 이동합니다: {t3_coords}")
            # mc.send_coords(t3_coords, speed, 0)
            # time.sleep(2)
            
            print(f"[INFO] to으로 이동합니다: {coords_o}")
            mc.send_coords(coords_o, speed, 0)
            time.sleep(2)


            place_pos = [89.91, -9.84, -95.8, 37.0, 4.48, -48.6]

            print("[INFO] 배치 위치로 이동...")
            mc.send_angles(place_pos, 20, 0)
            time.sleep(2)

            print("[INFO] 물체 놓기...")
            mc.set_gripper_value(100, 50)
            time.sleep(2)

            print("[INFO] 초기 위치로 복귀 중...")
            mc.send_angles(start_angles, speed, 0)
            time.sleep(2)

            break

