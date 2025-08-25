import numpy as np
import cv2
import glob
import pickle

# 체커보드 내부 코너 개수 (행, 열)
CHECKERBOARD = (8, 6)
square_size = 25.0  # mm

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 체커보드의 3D 좌표, prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= square_size

# Arrays to store object points and image points from all the images.
objpoints = []  # 3D points
imgpoints = []  # 2D points

images = glob.glob('/home/jetcobot/cam_calib/cam_image/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(200)

# ===== 캘리브레이션 실패 방지용 체크 =====
if not objpoints or not imgpoints:
    raise RuntimeError("체커보드 코너를 감지하지 못해 카메라 캘리브레이션에 실패했습니다.")

cv2.destroyAllWindows()

# 내부 파라미터 계산
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[:2][::-1], None, None)

img = cv2.imread(images[0])
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print("total error: ", tot_error/len(objpoints))
print("Camera Calibrated:", ret)
print("Camera Matrix:\n", mtx)
print("Distortion Coefficients:\n", dist)

# 결과 저장
# np 형태
np.savez('/home/jetcobot/cam_calib/calib_intrinsic.npz', ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# 파이썬 전용 저장, 피클
calibration_data = {
    'camera_matrix': mtx,
    'dist_coeffs': dist,
    'rvecs': rvecs,
    'tvecs': tvecs
}

with open('calib_data.pkl', 'wb') as f:
    pickle.dump(calibration_data, f)
## 다른 파일에서 피클 저장 불러오는 법
# import pickle

# with open('calib_data.pkl', 'rb') as f:
#     calibration_data = pickle.load(f)

# # 이제 딕셔너리에서 각 항목을 꺼내서 사용 가능
# mtx = calibration_data['camera_matrix']
# dist = calibration_data['dist_coeffs']
# rvecs = calibration_data['rvecs']
# tvecs = calibration_data['tvecs']
