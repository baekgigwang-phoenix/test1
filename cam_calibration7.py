import cv2
import imutils
import numpy as np
import time
from pymycobot.mycobot280 import MyCobot280
from scipy.spatial.transform import Rotation as R

# ----- JetCobot 연결 -----
mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
mc.thread_lock = True

# 초기 자세 및 그리퍼 열기
mc.send_coords([191.9, -62.7, 246.9, -179.17, 0.25, -40.51], 50, 0)
time.sleep(1)
mc.set_gripper_value(100, 50)
time.sleep(1)

# ----- 카메라 초기화 -----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(2)

# ----- ArUco 마커 탐지 설정 -----
# 어떤 마커 종류(예: AprilTag 36h11)를 쓸지 지정함.
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
# 탐지할 때 어떤 **세부 조건(임계값, 노이즈 허용 등)**을 쓸지 설정함.
aruco_params = cv2.aruco.DetectorParameters()
# 위 설정들을 기반으로 마커 탐지기 객체를 만듦.이제 이 detector 객체를 사용해서 영상에서 마커를 찾아낼 수 있음.
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)



# ----- 카메라 캘리브레이션 로드 -----
calib_data = np.load('/home/jetcobot/cam_calib/calib_intrinsic.npz')
K = calib_data['mtx'] 
# 카메라 매트릭스:   초점거리 fx,fy 
# [[fx   0  cx]   주점    cx,cy
#  [0   fy  cy]
#  [0    0   1]]
dist = calib_data['dist']
# 렌즈 왜곡 계수:
# [k1 k2 p1 p2 k3]



# 마커 한 변의 길이 (단위: m)
marker_length = 0.02
# objp는 단순상수값이지만 
# 카메라 좌표를 기준으로 마커의 pose(자세, 위치)를 알기 위한 기준점, 사각형 중심이(0,0,0)
# "이 마커 중심점을 기준으로 카메라의 위치와 방향을 추정할 때 쓰는 함수 
# success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)  --> rvec(위치), tvec(방향)
objp = np.array([
    [-marker_length/2,  marker_length/2, 0],
    [ marker_length/2,  marker_length/2, 0],
    [ marker_length/2, -marker_length/2, 0],
    [-marker_length/2, -marker_length/2, 0]
], dtype=np.float32)

print("[INFO] 카메라 영상 스트리밍 시작...")
last_print_time = 0



# ----- 마커 탐색 루프 -----
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 프레임을 읽을 수 없습니다.")
        break

    # frame = imutils.resize(frame, width=1000)

    corners, ids, _ = detector.detectMarkers(frame)
    # frame: 현재 카메라 영상 프레임 (OpenCV의 BGR 이미지)
    # detector: 앞서 생성한 cv2.aruco.ArucoDetector 객체
    # detectMarkers(): 프레임 속의 마커를 찾아냄
    # corners: 탐지된 각 마커의 4개 꼭짓점 좌표 (카메라 화면의 2D 픽셀 좌표)
    # corners 의 shape: [N][4][2] → 마커 개수 N개, 각 꼭짓점이 (x, y) 형태
    # ids: 탐지된 각 마커의 ID 번호 (정수 배열)
    #✅ 예: 마커가 3개 인식된 경우
    # 📦 ids:
    # ids = array([[7],
    #             [13],
    #             [42]], dtype=int32)
    # 마커의 ID 번호 3개 (각각 [7], [13], [42])
    # 보통 ids.flatten()을 하면: array([7, 13, 42])

    # 📦 corners: shape = (3, 4, 2)
    # corners = [
    #     [  # 첫 번째 마커 (ID: 7)의 꼭짓점들
    #         [x1_1, y1_1],
    #         [x1_2, y1_2],
    #         [x1_3, y1_3],
    #         [x1_4, y1_4]
    #     ],
    #     [  # 두 번째 마커 (ID: 13)
    #         [x2_1, y2_1],
    #         [x2_2, y2_2],
    #         [x2_3, y2_3],
    #         [x2_4, y2_4]
    #     ],
    #     [  # 세 번째 마커 (ID: 42)
    #         [x3_1, y3_1],
    #         [x3_2, y3_2],
    #         [x3_3, y3_3],
    #         [x3_4, y3_4]
    #     ]
    # ]
    # corners[0] ← ID 7의 꼭짓점 4개 (x, y)
    # corners[1] ← ID 13의 꼭짓점 4개
    # corners[2] ← ID 42의 꼭짓점 4개



    if ids is not None:
        ids = ids.flatten()
        # ids는 기본적으로 shape이 (N, 1)인 2D 배열인데, flatten()으로 1D로 만듦 → [7, 13, 42] 형태
        # corners와 함께 zip()으로 묶기 위해 필수

        for (marker_corner, marker_id) in zip(corners, ids):
            # [
            #     (corner1, 7),
            #     (corner2, 13),
            #     (corner3, 42)
            # ]
            pts = marker_corner.reshape((4, 2)).astype(int)
            # marker_corner: shape이 (1, 4, 2)일 수 있음 → (4, 2)로 reshape
            # 꼭짓점 4개를 2D 정수 좌표로 만듦
            # 예: [ [ [123, 78], [150, 79], [149, 105], [122, 104] ] ]에서 양끝괄호 1개삭제
            # pts = np.array([
            #     [123, 78],   # ← pts[0]
            #     [150, 79],   # ← pts[1]
            #     [149, 105],  # ← pts[2]
            #     [122, 104]   # ← pts[3]
            # ], dtype=int)
            # astype(int)은 OpenCV가 요구하는 픽셀 좌표 형식 때문 (정수여야 함)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            # 영상 프레임에 초록색 선(라인)을 그림
            # pts: 꼭짓점 4개 (시계/반시계 방향)
            # True: 마지막 점에서 첫 번째 점까지 닫힌 선으로 그림
            # (0, 255, 0): 초록색 (BGR 순서)
            # 2: 선의 두께
            # 즉, 마커의 윤곽선을 영상에 박스 형태로 그림
            cv2.putText(frame, str(marker_id), (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # 즉, 마커 ID 번호를 마커 옆에 표시합니다.
            # | 표현        | 값         | 의미                             
            # | pts[0]     | [123, 78]  | 첫 번째 꼭짓점 (왼쪽 위)                
            # | pts[0][0]  | 123        | x 좌표 (수평 위치, 오른쪽으로 옮기려면 +숫자픽셀) 
            # | pts[0][1]  | 78         | y 좌표 (수직 위치, 아래로 옮기려면 +숫자픽셀)   

            # 해당 마커 ID를 꼭짓점 pts[0] 근처에 표시
            # pts[0]는 보통 왼쪽 위 점
            # -10을 해서 살짝 위쪽에 글자가 뜨게 조정
            # FONT_HERSHEY_SIMPLEX: 일반적인 텍스트 폰트
            # 0.6: 글자 크기
            # (0, 255, 0): 초록색
            # 2: 글자 두께
            
        # 반복	marker_corner	marker_id
        # 1회차	  corners[0]	 7
        # 2회차	  corners[1]	 13
        # 3회차	  corners[2]	 42

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

# ----- 사용자로부터 추적할 마커 ID 입력 받기 -----
while True:
    try:
        target_id = int(input("[INPUT] 추적할 마커 ID를 입력하세요: "))
        break
    except ValueError:
        print("숫자 형태로 입력해주세요.")

print(f"[INFO] 마커 ID {target_id} 추적 시작")

# ----- 선택한 아루코 마커의 위치 계산 -----
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




# -----**4×4 pose 변환 행렬 T_m2c (마커 → 카메라 좌표계 변환)**을 만드는 과정-----
# 1. rvec → 3x3 회전 행렬 R
R, _ = cv2.Rodrigues(rvec)  # shape: (3, 3)

# 2. tvec → 3x1 이동 벡터
t = tvec.reshape(3, 1)      # shape: (3, 1)

# 3. T_m2c 구성 (4x4)
T_m2c = np.eye(4)           # 단위 행렬 생성
T_m2c[:3, :3] = R           # 좌측 상단에 R
T_m2c[:3, 3:] = t           # 우측 상단에 t

# 카메라 좌표계에서 본 마커의 pose
T_c2m = inverse(T_m2c)

# T_r2c 수동 측정(야메지롱)
# 카메라는 정면 방향 +Z, 카메라는 그립퍼보다 뒤로 50mm, 위로 40mm
t_r2c = np.array([[0],     # x: 그리퍼를 마주봤을 때 오른쪽
                  [40],     # y: 로봇 팔 기준 위로 올라가는 방향
                  [-50]])   # z: 그리퍼가 향하는 방향

# 카메라가 End-Effector와 완전히 평행하게 장착됐다면 → R = np.eye(3)
R_r2c = np.eye(3)
# 만약 카메라가 End-Effector에 대해 약간 기울어져 있다면, 직접 설정해야 합니다.
# Z축 기준으로 90도 회전 (카메라가 가로로 눕혀 있음)
# theta = np.radians(90)
# R_r2c = np.array([
#     [np.cos(theta), -np.sin(theta), 0],
#     [np.sin(theta),  np.cos(theta), 0],
#     [0,              0,             1]
# ])

# T_r2c 구성
T_r2c = np.eye(4)
T_r2c[:3, :3] = R_r2c
T_r2c[:3, 3] = (t_r2c / 1000.0).reshape(3)  # mm → m 변환
# OpenCV의 cv2.solvePnP() 같은 pose 추정 함수에서는, 
# 3D 위치(objp)는 "미터 단위"로 설정하는 것이 표준입니다.

# T_r2m 계산
T_r2m = T_r2c @ T_c2m

# 위치 (m → mm)
pos = T_r2m[:3, 3] * 1000  # mm

# 방향: 회전 행렬 → Euler angle (xyz, degree)
rot = R.from_matrix(T_r2m[:3, :3])
rx, ry, rz = rot.as_euler('xyz', degrees=True)

# 포즈 배열 구성 (JetCobot 좌표계 기준)
move_to_mark = [pos[0], pos[1], pos[2], rx, ry, rz]


# # ----- 좌표 변환 및 로봇 이동 -----

current_coords = mc.get_coords()
print("[INFO] 현재 로봇 좌표:", current_coords)

# print("[INFO] 이동할 로봇 포즈:", target_pose)
# mc.send_coords(target_pose, speed=50, mode=0)

target_coords = current_coords.copy()
target_coords[0] += move_to_mark[0]
target_coords[1] += move_to_mark[1]
target_coords[2] += move_to_mark[2]

print(f"[INFO] 타겟 좌표로 이동합니다: {target_coords}")
mc.send_coords(target_coords, 50, 0)
time.sleep(3)

print("[INFO] 그리퍼를 닫습니다.")
mc.set_gripper_value(0, 50)
time.sleep(1)

mc.send_coords([191.9, -62.7, 246.9, -179.17, 0.25, -40.51], 50, 0)
time.sleep(1)


# # ----- 좌표 변환 및 로봇 이동 -----
# x_cam, y_cam, z_cam = x, y, z
# x_robot = y_cam + 100   # x_cam → y축 + 오프셋
# y_robot = -x_cam        # y_cam → -x축
# z_robot = z_cam

# current_coords = mc.get_coords()
# print("[INFO] 현재 좌표:", current_coords)

# target_coords = current_coords.copy()
# target_coords[0] += x_robot
# target_coords[1] += y_robot
# target_coords[2] += -130

# print(f"[INFO] 타겟 좌표로 이동합니다: {target_coords}")
# mc.send_coords(target_coords, 50, 0)
# time.sleep(3)

# ----- 그리퍼 닫기 및 초기 위치 복귀 -----
# print("[INFO] 그리퍼를 닫습니다.")
# mc.set_gripper_value(0, 50)
# time.sleep(1)

# mc.send_coords([191.9, -62.7, 246.9, -179.17, 0.25, -40.51], 50, 0)
# time.sleep(1)
