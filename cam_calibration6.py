import argparse
import imutils
import time
import cv2
import sys
import numpy as np
from pymycobot.mycobot280 import MyCobot280
# JetCobot 연결
mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
mc.thread_lock = True
# JetCobot 초기 및 인식 자세
init_angles = [0, 118, -150, -37, -3, -50]
speed = 10
initial_angles = [0, 118, -150, -10, 0, -50]
speed = 10
pre_detect_angles = [0, -22.5, -22.5, -40, 0, -50]
speed = 10
# gripper 초기화
mc.set_gripper_state(0, 80)
time.sleep(1)
mc.send_angles(pre_detect_angles, 50)
time.sleep(3)
# 카메라 연결
cap = cv2.VideoCapture(0)  # 또는 /dev/jetcocam0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(2)
# 마커 딕셔너리 및 파라미터 설정
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# time.sleep(2)
last_print_time = 0  # 마지막 출력 시각 초기화
# ----------- 초기 탐색 단계 ------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 프레임을 읽을 수 없습니다.")
        break
    # frame = imutils.resize(frame, width=1000)
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topLeft = tuple(map(int, topLeft))
            topRight = tuple(map(int, topRight))
            bottomRight = tuple(map(int, bottomRight))
            bottomLeft = tuple(map(int, bottomLeft))
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #  60초마다만 출력
        if time.time() - last_print_time > 60:
            print(f"[INFO] 인식된 마커 ID들: {ids}")
            last_print_time = time.time()
    else:
        cv2.putText(frame, "마커를 인식할 수 없습니다.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "Press 'q' to input target ID", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
# 캘리브레이션 파일 로딩
data = np.load('/home/jetcobot/cam_calib/calib_intrinsic.npz')
K = data['mtx']
dist = data['dist']
# 마커 한 변의 길이 (단위: m)
marker_length = 0.02
# 마커의 3D 좌표계 설정
objp = np.array([
    [-marker_length/2,  marker_length/2, 0],
    [ marker_length/2,  marker_length/2, 0],
    [ marker_length/2, -marker_length/2, 0],
    [-marker_length/2, -marker_length/2, 0]
], dtype=np.float32)
# ----------- 사용자에게 추적할 마커 ID 입력 받기 --------------
while True:
    try:
        target_id = int(input("[INPUT] 추적할 마커 ID를 입력하세요: "))
        break
    except ValueError:
        print("숫자 형태로 입력해주세요.")
print(f"[INFO] 마커 ID {target_id} 추적을 시작합니다. 'q' 키로 종료.")
# ----------- 본격적인 추적 시작 루프 ----------------
last_pose_print_time = 0  # 좌표 출력 시간 기준
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 프레임을 읽을 수 없습니다.")
        break
    # frame = imutils.resize(frame, width=1000)
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        ids = ids.flatten()
        for i, marker_id in enumerate(ids):
            if marker_id != target_id:
                continue  # 다른 마커는 무시
            imgp = corners[i][0].astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
            if not success:
                continue
            # 위치 및 회전 정보 출력 (60초마다 1번만)
            current_time = time.time()
            if current_time - last_pose_print_time > 60:
                x, y, z = tvec.reshape(-1) * 1000
                rx, ry, rz = np.degrees(rvec.reshape(-1))
                print(f"[INFO] Marker ID {marker_id}")
                print(f"  위치 (mm): x={x:.1f}, y={y:.1f}, z={z:.1f}")
                print(f"  회전 (deg): rx={rx:.1f}, ry={ry:.1f}, rz={rz:.1f}")
                last_pose_print_time = current_time
            # 좌표축 시각화는 매 프레임
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, marker_length / 2)
            break  # 여러 마커 중 하나만 추적
    # 영상 출력
    cv2.imshow("AprilTag Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
# mc.send_coords([191.9, -62.7, 246.9, -179.17, 0.25, -40.51],50,0)
# # 좌표계확인
# current_coords = mc.get_coords()
# print("1번 위치 현재 좌표:", current_coords)
# # 목표 좌표 설정 (현재에서 조금 변경)
# target_coords = current_coords.copy()
# target_coords[0] -= 50 # X + 30mm
# target_coords[1] += 50 # Y - 30mm
# target_coords[2] -= 130 # Z - 50mm
# print(f"목표 좌표로 이동합니다: {target_coords}")
# mc.send_coords(target_coords, 50, 0)
# time.sleep(3)