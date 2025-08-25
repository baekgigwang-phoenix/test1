import cv2
import numpy as np
import time
from pymycobot.mycobot280 import MyCobot280


# ===== 전역 설정 =====
MARKER_LENGTH = 0.02  # m
SPEED = 30

INITIAL_ANGLES = [-3.77, 109.07, -133.41, 21.44, 5.88, -44.91]
START_ANGLES = [-2.02, 10.98, -130.78, 125, 1.31, -46.31]
PLACE_ANGLES = [89.91, -9.84, -95.8, 37.0, 4.48, -48.6]


# ===== 1. 로봇 연결 및 초기화 =====
def init_robot():
    mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
    mc.thread_lock = True
    print("[INFO] 로봇이 연결되었습니다.")

    mc.send_angles(INITIAL_ANGLES, SPEED)
    mc.set_gripper_value(100, SPEED)
    time.sleep(1)
    return mc


# ===== 2. 카메라 및 마커 탐지기 초기화 =====
def init_camera_and_detector():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    calib_data = np.load('/home/jetcobot/cam_calib/calib_intrinsic.npz')
    K, dist = calib_data['mtx'], calib_data['dist']

    objp = np.array([
        [-MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
        [ MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
        [ MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
        [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0]
    ], dtype=np.float32)

    return cap, detector, K, dist, objp


# ===== 3. 마커 탐지 및 ID 선택 =====
def select_marker_id(cap, detector):
    print("[INFO] 카메라 영상 스트리밍 시작...")
    last_print_time = 0

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

    cv2.destroyWindow("Marker Detection")

    while True:
        try:
            target_id = int(input("[INPUT] 추적할 마커 ID를 입력하세요: "))
            return target_id
        except ValueError:
            print("숫자로 입력해주세요.")


# ===== 4. 마커 좌표 계산 =====
def get_marker_coords(cap, detector, target_id, objp, K, dist):
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] 카메라 프레임 수신 실패")
        return None

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        print("[ERROR] 마커가 감지되지 않았습니다.")
        return None

    ids = ids.flatten()
    for i, marker_id in enumerate(ids):
        if marker_id != target_id:
            continue

        imgp = corners[i][0].astype(np.float32)
        success, _, tvec = cv2.solvePnP(objp, imgp, K, dist)
        if not success:
            print("[ERROR] solvePnP 실패")
            return None

        x_c, y_c, z_c = [round(float(val), 2) for val in tvec.reshape(-1) * 1000]
        z_c -= 55.37
        return x_c, y_c, z_c

    print("[ERROR] 선택한 마커 ID를 찾지 못했습니다.")
    return None


# ===== 5. 로봇 이동 및 물체 조작 =====
def move_and_pick(mc, x_c, y_c, z_c):
    coords_o = mc.get_coords()
    if coords_o is None:
        print("[ERROR] 로봇 좌표 수신 실패")
        return

    # 접근 위치
    t1_coords = coords_o.copy()
    t1_coords[0] += 10
    t1_coords[1] -= x_c
    t1_coords[2] -= y_c
    t1_coords[2] += 80
    t1_coords = [round(val, 2) for val in t1_coords]

    print(f"[INFO] t1으로 이동: {t1_coords}")
    mc.send_coords(t1_coords, SPEED, 0)
    time.sleep(2)

    # 잡기 위치
    t2_coords = t1_coords.copy()
    t2_coords[0] += 240
    t2_coords[2] += 20
    t2_coords = [round(val, 2) for val in t2_coords]

    print(f"[INFO] t2으로 이동: {t2_coords}")
    mc.send_coords(t2_coords, SPEED, 0)
    time.sleep(3)

    # 잡기
    print("[INFO] 물건 잡기")
    mc.set_gripper_value(0, 50)
    time.sleep(2)

    # 원래 위치 복귀
    print(f"[INFO] 원래 위치로 이동: {coords_o}")
    mc.send_coords(coords_o, SPEED, 0)
    time.sleep(2)

    # 배치 위치로 이동
    print("[INFO] 배치 위치로 이동...")
    mc.send_angles(PLACE_ANGLES, 20, 0)
    time.sleep(2)

    # 물건 놓기
    print("[INFO] 물체 놓기...")
    mc.set_gripper_value(100, 50)
    time.sleep(2)

    # 시작 위치 복귀
    print("[INFO] 초기 위치로 복귀 중...")
    mc.send_angles(START_ANGLES, SPEED, 0)
    time.sleep(2)


# ===== 메인 실행 =====
def main():
    mc = init_robot()
    cap, detector, K, dist, objp = init_camera_and_detector()
    target_id = select_marker_id(cap, detector)

    coords = get_marker_coords(cap, detector, target_id, objp, K, dist)
    cap.release()
    cv2.destroyAllWindows()

    if coords:
        x_c, y_c, z_c = coords
        move_and_pick(mc, x_c, y_c, z_c)


if __name__ == "__main__":
    main()
