# 로봇팔을 초기측정위치로 이동 후, 
# 캘리브레이션된 카메라 화면으로 물건을 인식하고
# 정면,좌,우,위,아래 방향에서의 사진을 촬영하여 저장한다.

import cv2
import imutils
import numpy as np
import time
import os
from pymycobot.mycobot280 import MyCobot280
from scipy.spatial.transform import Rotation as R

# ----- npz 파일에서 보정 데이터 불러오는 함수 -----
def load_calibration_from_npz(npz_path):
    data = np.load(npz_path)
    return data['mtx'], data['dist']

# ----- 왜곡 보정 함수 -----
def undistort_frame(frame, mtx, dist):
    # 프레임 크기 가져오기
    h, w = frame.shape[:2]
    # 최적의 카메라 행렬 구하기
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # 왜곡 보정
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    # ROI로 이미지 자르기
    x, y, w, h = roi
    if all(v > 0 for v in [x, y, w, h]):
        undistorted = undistorted[y:y+h, x:x+w]
    return undistorted[y:y+h, x:x+w]

# ----- 촬영 전 카메라 이동 함수 -----
def move_location(saved_count):
    # 정면
    if saved_count == 0:
        mc.send_coords([191.9, -62.7, 246.9, -179.17, 0.25, -40.51], 50, 0)
        time.sleep(1)
    # 좌측
    elif saved_count == 1:
        mc.send_coords([176.4, 51.6, 157.1, 155.71, -20.35, -35.9], 50, 0)
        time.sleep(1)
    # 우측
    elif saved_count == 2:
        mc.send_coords([174.6, -146.7, 163.7, -151.04, 16.53, -34.4], 50, 0)
        time.sleep(1)
    # 위
    elif saved_count == 3:
        mc.send_coords([225.3, -57.5, 191.7, 175.85, 11.53, -39.34], 50, 0)
        time.sleep(1)
    # 아래
    elif saved_count == 4:
        mc.send_coords([64.9, -59.9, 188.3, -158.41, -22.1, -37.91], 50, 0)
        time.sleep(1)



# ----- JetCobot 연결 -----
mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
mc.thread_lock = True

# ---- 초기 자세 및 그리퍼 열기 ----
mc.send_coords([191.9, -62.7, 246.9, -179.17, 0.25, -40.51], 50, 0)
time.sleep(1)
mc.set_gripper_value(100, 50)
time.sleep(1)

# 이미 캘리브레이션 파일이 있는지 확인
if os.path.exists('/home/jetcobot/cam_calib/calib_intrinsic.npz'):
    print("Loading existing calibration data...")
else:
    raise FileNotFoundError("캘리브레이션 파일(camera_calibration.pkl)이 존재하지 않습니다. 먼저 생성하세요.")

# npz 파일에서 보정 데이터 불러오기
npz_path = '/home/jetcobot/cam_calib/calib_intrinsic.npz'
mtx, dist = load_calibration_from_npz(npz_path)

# 카메라 실행
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("카메라를 열 수 없습니다.")

print("보정된 영상 스트림을 시작합니다. 's'키로 최대 다섯장 촬영, 'q'는 촬영 중지")

saved_count = 0  # 저장된 이미지 개수
max_photos = 5   # 최대 촬영 수

# # ----- 카메라 초기화 -----
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corrected_frame = undistort_frame(frame, mtx, dist)
    cv2.imshow("Undistorted Camera View", corrected_frame)

    key = cv2.waitKey(1) & 0xFF

    # 사진 저장
    if key == ord('s') and saved_count < max_photos:
        move_location(saved_count)
        time.sleep(2)
        filename = f"/home/jetcobot/captured_marker_img/captured_marker_{saved_count+1}.jpg"
        cv2.imwrite(filename, corrected_frame)
        print(f"[저장 완료] {filename}")
        saved_count += 1

        if saved_count == max_photos:
            print("5장의 사진을 모두 저장했습니다. 프로그램을 종료합니다.")
            break


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

time.sleep(1)
mc.send_coords([191.9, -62.7, 246.9, -179.17, 0.25, -40.51], 50, 0)


