import os
import time
import threading
import math
import cv2
import numpy as np
from pymycobot.mycobot280 import MyCobot280
from pymycobot.genre import Angle, Coord
# from ultralytics import YOLO

# ------------------------- MyCobot 초기화 -------------------------
mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
mc.thread_lock = True
print("로봇이 연결되었습니다.")

# 현재 상태 출력
angles = mc.get_angles()
coords = mc.get_coords()
encoders = mc.get_encoders()
radians = mc.get_radians()
print("현재 각도:", angles)
print("현재 좌표:", coords)
print("인코더:", encoders)
print("라디안:", radians)

# 각 관절 범위 출력
ANGLE_MIN = [-168, -135, -150, -145, -165, -180, 0]
ANGLE_MAX = [168, 135, 150, 145, 165, 180, 100]
for i in range(7):
    print(f"관절 {i+1}: {ANGLE_MIN[i]} ~ {ANGLE_MAX[i]}도")

# 0 위치로 이동
initial_angles = [0, 0, 0, 0, 0, -45]
speed = 50
print("로봇을 0 위치로 리셋합니다.")
mc.send_angles(initial_angles, speed)
mc.set_gripper_value(100, speed)  # 그리퍼 열기
time.sleep(2)
print("리셋 완료")

# 1 위치로 이동
initial_angles = [0, 0, 0, -60, 0, -45]
speed = 50
print("로봇을 1 위치로 리셋합니다.")
mc.send_angles(initial_angles, speed)
mc.set_gripper_value(100, speed)  # 그리퍼 열기
time.sleep(2)
print("리셋 완료")

# # ------------------------- OpenCV 실시간 인식 ver1 -------------------------
# # 카메라 열기
# cap = cv2.VideoCapture(0)  # USB 웹캠

# # HSV 범위 설정 (빨간색 예시)
# lower_red = np.array([0, 100, 100])
# upper_red = np.array([10, 255, 255])

# # 카메라 해상도 (640x480)
# frame_width = 640
# frame_height = 480

# # 화면 중앙 좌표
# cx_target = frame_width // 2
# cy_target = frame_height // 2


# # start_time = time.time()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # # 현재 시간이 시작 시간으로부터 5초가 경과했는지 확인
#     # elapsed_time = time.time() - start_time

#     # # 5초가 지나면 루프 종료
#     # if elapsed_time > 5:
#     #     print("5초가 지나서 인식을 종료합니다.")
#     #     time.sleep(5)
#     #     break
    
#     # 색상 인식 처리
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, lower_red, upper_red)
#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours: #5초만 빨간색 인식해서 cx, cy값 구하는 것으로 해보자
#         area = cv2.contourArea(cnt)
#         if area < 500:
#             continue
#         x, y, w, h = cv2.boundingRect(cnt)
#         cx = x + w // 2
#         cy = y + h // 2

#          # 시각화
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
#         cv2.putText(frame, f"({cx},{cy})", (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#         cv2.circle(frame, (cx_target, cy_target), 5, (0, 0, 255), -1)
        
#         # x, y의 차이를 계산 (오차)
#         dx = cx_target - cx
#         dy = cy_target - cy

#         print(f"인식된 빨간 물체 중심 좌표: ({cx}, {cy})")
#         print(f"화면 중심 좌표: ({cx_target}, {cy_target})")
#         print(f"오차: ({dx}, {dy})")
#         # time.sleep(1)
    
        
#     # 결과 화면 출력
#     cv2.imshow("Camera View", frame)
#     cv2.imshow("Red Mask", mask)
    
#     # # q 누르면 루프 종료
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break

# # 최종 측정값
# print(f"인식된 빨간 물체 중심 좌표: ({cx}, {cy})")
# print(f"화면 중심 좌표: ({cx_target}, {cy_target})")
# print(f"오차: ({dx}, {dy})")


# 카메라 화면을 실시간으로 업데이트할 함수
def show_camera():
    cap = cv2.VideoCapture(0)  # USB 웹캠

    # HSV 범위 설정 (빨간색 예시)
    lower_red = np.array([170, 100, 100])
    upper_red = np.array([180, 255, 255])

    # 카메라 해상도 (640x480)
    frame_width = 640
    frame_height = 480

    # 화면 중앙 좌표
    cx_target = frame_width // 2
    cy_target = frame_height // 2

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 현재 시간이 시작 시간으로부터 5초가 경과했는지 확인
        elapsed_time = time.time() - start_time

        # 5초가 지나면 루프 종료
        if elapsed_time > 5:
            print("5초가 지나서 인식을 종료합니다.")
            time.sleep(5)
            break

        # 색상 인식 처리
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours: 
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

            # 시각화
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(frame, f"({cx},{cy})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.circle(frame, (cx_target, cy_target), 5, (0, 0, 255), -1)

        dx = cx_target - cx
        dy = cy_target - cy

        # 결과 화면 출력
        cv2.imshow("Camera View", frame)
        cv2.imshow("Red Mask", mask)

    # 최종 측정값
    print(f"인식된 빨간 물체 중심 좌표: ({cx}, {cy})")
    print(f"화면 중심 좌표: ({cx_target}, {cy_target})")
    print(f"오차: ({dx}, {dy})")

    

    cap.release()
    cv2.destroyAllWindows()


# 카메라 화면을 별도의 스레드에서 실행
def move_point():

    camera_thread = threading.Thread(target=show_camera)
    camera_thread.start()

    if dx > -20 and dx < 20 and dy > -20 and dy < 20
        return

    while not (-20 < dx < 20 and -20 < dy < 20):
        if not ret:
            break

        camera_thread = threading.Thread(target=show_camera)
        camera_thread.start()
        time.sleep(5)

        if dx > 0 and dy > 0:  
            current_coords = mc.get_coords()
            # 목표 좌표 설정 (현재에서 조금 변경)
            target_coords = current_coords.copy()
            target_coords[0] -= 30 # X + 30mm
            target_coords[1] -= 30 # Y - 30mm
            mc.send_coords(target_coords, 50, 0)
            time.sleep(1)

        elif dx < 0 and dy > 0:  
            current_coords = mc.get_coords()
            # 목표 좌표 설정 (현재에서 조금 변경)
            target_coords = current_coords.copy()
            target_coords[0] += 30 # X + 30mm
            target_coords[1] -= 30 # Y - 30mm
            mc.send_coords(target_coords, 50, 0)
            time.sleep(1)
        
        elif dx > 0 and dy < 0: 
            current_coords = mc.get_coords()
            # 목표 좌표 설정 (현재에서 조금 변경)
            target_coords = current_coords.copy()
            target_coords[0] -= 30 # X + 30mm
            target_coords[1] += 30 # Y - 30mm
            mc.send_coords(target_coords, 50, 0)
            time.sleep(1)
        
        elif dx < 0 and dy < 0:  
            current_coords = mc.get_coords()
            # 목표 좌표 설정 (현재에서 조금 변경)
            target_coords = current_coords.copy()
            target_coords[0] -= 30 # X + 30mm
            target_coords[1] -= 30 # Y - 30mm
            mc.send_coords(target_coords, 50, 0)
            time.sleep(1)




def move_robot(target_coords):
    mc.send_coords(target_coords, 50, 0)
    time.sleep(2)


time.sleep(3)


# # dx, dy에 따라 로봇 1차 이동
# if dx > 0 and dy > 0 and dx < 20 and dy < 30:  # 위치 1로 이동, 왼쪽 멀리
#     target_coords = [205.1, -54.8, 98.8, -176.57, -3.77, -38.51]
#     move_robot(target_coords)
    
# elif dx > 0 and dy > 0:  # 위치 1로 이동, 왼쪽 멀리
#     target_coords = [196.8, 2.9, 165.1, -173.16, -4.96, -48.08]
#     move_robot(target_coords)

# elif dx < 0 and dy > 0:  # 위치 2로 이동 오른쪽 멀리
#     target_coords = [181.1, -93.9, 198.0, -175.93, -3.83, -48.61]
#     move_robot(target_coords)
   
# elif dx > 0 and dy < 0:  # 위치 3으로 이동 왼족 짧게
#     target_coords = [129.8, -17.6, 157.3, -175.3, -5.3, -42.03]
#     move_robot(target_coords)
  
# elif dx < 0 and dy < 0:  # 위치 4로 이동 오른쪽 짧게
#     target_coords = [126.7, -92.6, 164.6, -179.43, -2.69, -49.87]
#     move_robot(target_coords)
 

# # 로봇을 지정된 위치로 이동
# mc.send_angles(initial_angles, speed)


# # dx, dy에 따라 로봇 2차 이동
# if dx > 0 and dy > 0 and dx < 20 and dy < 30:

#     move_point()

# elif dx > 0 and dy > 0:  # 위치 1로 이동, 왼쪽 멀리

#     move_point()

# elif dx < 0 and dy > 0:  # 위치 2로 이동 오른쪽 멀리

#     move_point()

# elif dx > 0 and dy < 0:  # 위치 3으로 이동 왼족 짧게

#     move_point()

        
# elif dx < 0 and dy < 0:  # 위치 4로 이동 오른쪽 짧게

#     move_point()







# cap.release()
# cv2.destroyAllWindows()



# ------------------------- OpenCV 실시간 인식 ver2 -------------------------
# # ArUco 마커 설정
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# parameters = cv2.aruco.DetectorParameters()

# # YOLOv8 모델 로드
# model = YOLO('yolov8n.pt')

# # 마커 ID -> 이름 매핑 (예: a~g)
# id_to_name = {
#     0: 'A', 1: 'B', 2: 'C',
#     3: 'D', 4: 'E', 5: 'F', 6: 'G'
# }

# # 카메라 열기 (0 또는 /dev/video0 등)
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("카메라 열기 실패")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("프레임 읽기 실패")
#         break

#     # ✅ YOLO 추론
#     results = model.predict(source=frame, save=False, conf=0.5, imgsz=640)
#     annotated_frame = results[0].plot()

#     # ✅ ArUco 마커 탐지
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

#     if ids is not None:
#         for i, corner in zip(ids, corners):
#             i = int(i[0])  # numpy -> int
#             label = id_to_name.get(i, f"ID:{i}")

#             # 마커 중심 좌표 계산
#             center = corner[0].mean(axis=0).astype(int)
#             cv2.putText(annotated_frame, label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#             cv2.polylines(annotated_frame, [corner.astype(int)], True, (0,255,255), 2)

#     # ✅ 출력
#     cv2.imshow("YOLO + ArUco Detection", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



