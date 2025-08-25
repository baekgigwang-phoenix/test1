import cv2
import numpy as np
import pickle

# 데이터 가져오기
def load_calibration_from_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        calib_data = pickle.load(f)
    return calib_data['camera_matrix'], calib_data['dist_coeffs']

# 캘리브레이션 
def undistort_frame(frame, mtx, dist):
    # 프레임 크기 가져오기
    h, w = frame.shape[:2]
    # 최적의 카메라 행렬 구하기
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # 왜곡 보정
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    # ROI로 이미지 자르기
    x, y, w, h = roi
    return undistorted[y:y+h, x:x+w]

def detect_aruco_and_estimate_pose(frame, mtx, dist, marker_size = 0.02):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(frame)

    marker_length = 0.02

    if corners:
        for corner in corners:
            corner = np.array(corner).reshape((4, 2))
            marker_3d = np.array([             
                [-marker_length/2,  marker_length/2, 0],
                [ marker_length/2,  marker_length/2, 0],
                [ marker_length/2, -marker_length/2, 0],
                [-marker_length/2, -marker_length/2, 0]
                ], dtype=np.float32)
            #     [0, 0, 0],
            #     [0, marker_size, 0],
            #     [marker_size, marker_size, 0],
            #     [marker_size, 0, 0]
            # ], dtype=np.float32).reshape((4, 1, 3))

            ret, rvec, tvec = cv2.solvePnP(marker_3d, corner, mtx, dist)
            if ret:
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, marker_size / 2)

                pos_text = f"Pos: ({tvec[0][0]:.1f}, {tvec[1][0]:.1f}, {tvec[2][0]:.1f}) mm"
                rot_text = f"Rot: ({np.rad2deg(rvec[0][0]):.1f}, {np.rad2deg(rvec[1][0]):.1f}, {np.rad2deg(rvec[2][0]):.1f}) deg"
                cv2.putText(frame, pos_text, (int(corner[0][0]) - 10, int(corner[0][1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, rot_text, (int(corner[0][0]) - 10, int(corner[0][1]) + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame


# 1. 캘리브레이션 불러오기
pkl_path = '/home/jetcobot/calib_data.pkl'
mtx, dist = load_calibration_from_pickle(pkl_path)

# 2. 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("카메라를 열 수 없습니다.")

print("보정 + 아루코 마커 검출을 시작합니다. 'q' 키로 종료하세요.")

# 3. 반복
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3-1. 보정
    undistorted = undistort_frame(frame, mtx, dist)

    # 3-2. 마커 검출 + 포즈 추정
    result_frame = detect_aruco_and_estimate_pose(undistorted, mtx, dist)

    # 3-3. 화면 출력
    cv2.imshow("ArUco Pose Estimation", result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. 종료
cap.release()
cv2.destroyAllWindows()
