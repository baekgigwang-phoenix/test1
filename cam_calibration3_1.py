# 캘리브레이션된 카메라 화면으로 하나의 마커 인식(마커가 하나만 있다 가정) 후,
# 캘리브레이션된 상태로 촬영
import cv2
import numpy as np
import os
import glob


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

# ----- 메인 루프 -----
if __name__ == "__main__":
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

    print("보정된 영상 스트림을 시작합니다. 'q' 키로 종료하세요.")

    saved_count = 0  # 저장된 이미지 개수
    max_photos = 5   # 최대 촬영 수

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corrected_frame = undistort_frame(frame, mtx, dist)
        cv2.imshow("Undistorted Camera View", corrected_frame)

        key = cv2.waitKey(1) & 0xFF

        # 사진 저장
        if key == ord('s') and saved_count < max_photos:
            filename = f"captured_marker_{saved_count+1}.jpg"
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


