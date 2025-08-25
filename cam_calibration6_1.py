import cv2
import time
import numpy as np
from pymycobot.mycobot280 import MyCobot280

# 로봇 초기화 및 자세 설정
def initialize_robot():
    mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
    mc.thread_lock = True
    mc.set_gripper_state(0, 80)
    time.sleep(1)
    pre_detect_angles = [0, -22.5, -22.5, -40, 0, -50]
    mc.send_angles(pre_detect_angles, 50)
    time.sleep(3)
    return mc

# 카메라 설정 및 객체 반환
def initialize_camera(width=640, height=480):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    time.sleep(2)
    return cap

# 카메라 내참 파라미터 불러오기
def load_calibration(file_path):
    data = np.load(file_path)
    return data['mtx'], data['dist']

# 마커 탐지
def detect_markers(frame, detector):
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        return corners, ids.flatten()
    return [], []

# 마커 시각화
def draw_marker(frame, corners, ids):
    for (markerCorner, markerID) in zip(corners, ids):
        pts = markerCorner.reshape((4, 2)).astype(int)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, str(markerID), (pts[0][0], pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 마커 추적 루프
def track_target_marker(cap, detector, mtx, dist, objp, target_id, marker_length):
    last_pose_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임을 읽을 수 없습니다.")
            break
        corners, ids = detect_markers(frame, detector)
        if ids is not None:
            for i, marker_id in enumerate(ids):
                if marker_id != target_id:
                    continue
                imgp = corners[i][0].astype(np.float32)
                success, rvec, tvec = cv2.solvePnP(objp, imgp, mtx, dist)
                if success:
                    current_time = time.time()
                    if current_time - last_pose_time > 60:
                        x, y, z = tvec.reshape(-1) * 1000
                        rx, ry, rz = np.degrees(rvec.reshape(-1))
                        print(f"[INFO] Marker ID {marker_id}")
                        print(f"  위치 (mm): x={x:.1f}, y={y:.1f}, z={z:.1f}")
                        print(f"  회전 (deg): rx={rx:.1f}, ry={ry:.1f}, rz={rz:.1f}")
                        last_pose_time = current_time
                    cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, marker_length / 2)
                    break
        cv2.imshow("AprilTag Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# 전체 흐름 실행
def main():
    print("[INFO] 로봇 초기화 중...")
    mc = initialize_robot()

    print("[INFO] 카메라 초기화 중...")
    cap = initialize_camera()

    print("[INFO] 마커 탐지기 설정...")
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(arucoDict, cv2.aruco.DetectorParameters())

    print("[INFO] 초기 영상 스트림 시작...")
    last_print_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 프레임을 읽을 수 없습니다.")
            break
        corners, ids = detect_markers(frame, detector)
        if ids is not None:
            draw_marker(frame, corners, ids)
            if time.time() - last_print_time > 60:
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
    cv2.destroyAllWindows()

    # 마커 ID 입력
    while True:
        try:
            target_id = int(input("[INPUT] 추적할 마커 ID를 입력하세요: "))
            break
        except ValueError:
            print("숫자 형태로 입력해주세요.")
    print(f"[INFO] 마커 ID {target_id} 추적을 시작합니다. 'q' 키로 종료.")

    # 캘리브레이션 및 마커 설정
    mtx, dist = load_calibration('/home/jetcobot/cam_calib/calib_intrinsic.npz')
    marker_length = 0.02  # 20mm
    objp = np.array([
        [-marker_length/2,  marker_length/2, 0],
        [ marker_length/2,  marker_length/2, 0],
        [ marker_length/2, -marker_length/2, 0],
        [-marker_length/2, -marker_length/2, 0]
    ], dtype=np.float32)

    # 마커 추적 시작
    track_target_marker(cap, detector, mtx, dist, objp, target_id, marker_length)

if __name__ == "__main__":
    main()
