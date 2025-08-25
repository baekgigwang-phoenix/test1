import cv2
import imutils
import numpy as np
import time
from pymycobot.mycobot280 import MyCobot280
from scipy.spatial.transform import Rotation as R

# JetCobot 연결
mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
mc.thread_lock = True
print("로봇이 연결되었습니다.")

# 0 위치로 이동
initial_angles = [0, 0, 0, 0, 0, -45]
speed = 50
print("로봇을 초기 위치로 리셋합니다.")
mc.send_angles(initial_angles, speed)
mc.set_gripper_value(100, speed)  # 그리퍼 열기
time.sleep(1)
print("리셋 완료")

# 시작 위치로 이동
start_angles = [-2.37, -128.75, 144.84, -14.67, 0.43, -44.73]
print("로봇을 시작 위치로 리셋합니다.")
mc.send_angles(start_angles, speed)
mc.set_gripper_value(100, speed)  # 그리퍼 열기
time.sleep(1)
print("리셋 완료")

# 카메라 초기화 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(1)

# ArUco 마커 탐지 설정 
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# 카메라 캘리브레이션 로드
calib_data = np.load('/home/jetcobot/cam_calib/calib_intrinsic.npz')
K = calib_data['mtx'] 
dist = calib_data['dist']

# 마커 한 변의 길이 (단위: m)
marker_length = 0.02

# success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist) 
#  --> rvec(위치), tvec(방향)
objp = np.array([
    [-marker_length/2,  marker_length/2, 0],
    [ marker_length/2,  marker_length/2, 0],
    [ marker_length/2, -marker_length/2, 0],
    [-marker_length/2, -marker_length/2, 0]
], dtype=np.float32)

print("[INFO] 카메라 영상 스트리밍 시작...")
last_print_time = 0

# 마커 종류 탐색
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 프레임을 읽을 수 없습니다.")
        break
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        ids = ids.flatten()
        for (marker_corner, marker_id) in zip(corners, ids):
            pts = marker_corner.reshape((4, 2)).astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(frame, str(marker_id), (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if time.time() - last_print_time > 10:
            print(f"[INFO] 인식된 마커 ID들: {ids}")
            last_print_time = time.time()
    else:
        cv2.putText(frame, "마커를 인식할 수 없습니다.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, "Press 'q' to input target ID", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyWindow("Frame")

# 사용자로부터 추적할 마커 ID 입력 받기
while True:
    try:
        target_id = int(input("[INPUT] 추적할 마커 ID를 입력하세요: "))
        break
    except ValueError:
        print("숫자 형태로 입력해주세요.")

print(f"[INFO] 마커 ID {target_id} 추적 시작")

# 선택한 아루코 마커의 위치 계산
ret, frame = cap.read()
if ret:
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        ids = ids.flatten()
        for i, marker_id in enumerate(ids):
            if marker_id != target_id:
                continue

            imgp = corners[i][0].astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
            # rvec = [[0.1],
            #         [0.5],
            #         [0.0]]
            # tvec = [[0.4],
            #         [0.55],
            #         [0.04]]

            if not success:
                continue

            x, y, z = tvec.reshape(-1) * 1000  # mm 단위
            rx, ry, rz = np.degrees(rvec.reshape(-1))

            print(f"[INFO] Marker ID {marker_id}")
            print(f"  위치 (mm): x={x:.1f}, y={y:.1f}, z={z:.1f}")
            print(f"  회전 (deg): rx={rx:.1f}, ry={ry:.1f}, rz={rz:.1f}")

            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, marker_length / 2)
            cv2.imshow("AprilTag Detection", frame)
            cv2.waitKey(1000)
            break

cv2.destroyAllWindows()

# 4×4 pose 변환 행렬
# 1. rvec → 3x3 회전 행렬 R
R, _ = cv2.Rodrigues(rvec)  # shape: (3, 3)

# 2. tvec → 3x1 이동 벡터
t = tvec.reshape(3, 1)      # shape: (3, 1) or "tvec.flatten()"

# 3. T_m2c 구성 (4x4)
T_m2c = np.eye(4)           # 단위 행렬 생성
T_m2c[:3, :3] = R           # 좌측 상단에 R
T_m2c[:3, 3:] = t           # 우측 상단에 t

T_m2c[:3, 3] += np.array([0, -0.04, -0.05])

T_m2ee = T_m2c

coords = mc.get_coords()
t_ee2r = np.array(coords[:3], dtype=np.float32)/1000.0  # mm --> m,    t_ee2r = np.array([0.150, -0.100, 0.250]) 
r_ee2r = R.from_euler('zyx', coords[3:], degrees=True)
R_ee2r = r_ee2r.as_matrix()

# 변환 행렬 구성
T_ee2r = np.eye(4)
T_ee2r[:3, :3] = R_ee2r
T_ee2r[:3, 3] = t_ee2r  

T_m2r = T_ee2r @ T_m2ee

# 최종 위치 (로봇 기준, mm 단위로 변환)
target_pos = T_m2r[:3, 3] * 1000
target_rot = R.from_matrix(T_m2r[:3, :3]).as_euler('zyx', degrees=True)

# 자세 안정화를 위해 필요시 회전 일부 고정 (예: z축만 따라가고 나머지는 고정)
rx, ry, rz = -179.17, 0.25, float(target_rot[2])  # 예시

# 마커 위로 접근 (상승 높이 100mm)
approach_z = target_pos[2] + 100

print("[INFO] 마커 상단으로 이동 중...")
mc.send_coords([target_pos[0], target_pos[1], approach_z, rx, ry, rz], 50, 0)
time.sleep(2)

print("[INFO] 마커 중심으로 하강 중...")
mc.send_coords([target_pos[0], target_pos[1], target_pos[2], rx, ry, rz], 50, 0)
time.sleep(2)

print("[INFO] 물체 집기 동작...")
mc.set_gripper_value(30, 50)  # 0 = 완전히 닫힘
time.sleep(1)

# 놓을 위치로 이동 
place_pos = [target_pos[0] - 60, target_pos[1], target_pos[2]]  # x축으로 60mm 이동

print("[INFO] 배치 위치로 이동...")
mc.send_coords([place_pos[0], place_pos[1], place_pos[2] + 50, rx, ry, rz], 50, 0)
time.sleep(2)

mc.send_coords([place_pos[0], place_pos[1], place_pos[2], rx, ry, rz], 50, 0)
time.sleep(2)

print("[INFO] 물체 놓기...")
mc.set_gripper_value(100, 50)  # 열기
time.sleep(1)

print("[INFO] 초기 위치로 복귀...")
mc.send_coords(start_coords, speed, 0)
time.sleep(2)

