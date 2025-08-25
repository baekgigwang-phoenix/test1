## 보정 잘되었는지 카메라 비교용


import cv2
import numpy as np
import os
import glob
import pickle

def live_video_correction(calibration_data):
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['dist_coeffs']
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 크기 가져오기
        h, w = frame.shape[:2]
        
        # 최적의 카메라 행렬 구하기
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # 왜곡 보정
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        # ROI로 이미지 자르기
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            dst = dst[y:y+h, x:x+w]
        
        # 원본과 보정된 이미지를 나란히 표시
        original = cv2.resize(frame, (640, 480))
        corrected = cv2.resize(dst, (640, 480))
        combined = np.hstack((original, corrected))
        
        # 결과 표시
        cv2.imshow('Original | Corrected', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 이미 캘리브레이션 파일이 있는지 확인
    if os.path.exists('calib_data.pkl'):
        print("Loading existing calibration data...")
        with open('calib_data.pkl', 'rb') as f:
            calibration_data = pickle.load(f)
    else:
        raise FileNotFoundError("캘리브레이션 파일(camera_calibration.pkl)이 존재하지 않습니다. 먼저 생성하세요.")

   
    print("Starting live video correction...")
    live_video_correction(calibration_data)


