import cv2
import numpy as np
import os
import glob
import pickle

# 함수를 이용하여 캘리브리션된 카메라화면 불러오기 예

def load_calibration_from_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        calib_data = pickle.load(f)
    return calib_data['camera_matrix'], calib_data['dist_coeffs']

def undistort_frame(frame, mtx, dist):
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    return undistorted[y:y+h, x:x+w]

if __name__ == "__main__":
    # 피클 파일에서 보정 데이터 불러오기
    pkl_path = '/home/jetcobot/calib_data.pkl'
    mtx, dist = load_calibration_from_pickle(pkl_path)

    # 카메라 실행
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("카메라를 열 수 없습니다.")

    print("보정된 영상 스트림을 시작합니다. 'q' 키로 종료하세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corrected_frame = undistort_frame(frame, mtx, dist)
        cv2.imshow("Undistorted Camera View", corrected_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


