import cv2
import numpy as np
import time
from pymycobot.mycobot280 import MyCobot280
from scipy.spatial.transform import Rotation as R

import argparse
import imutils
import sys


# ----- 1. JetCobot 연결 및 초기 위치 설정 -----
mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
mc.thread_lock = True
print("[INFO] 로봇이 연결되었습니다.")

speed = 30

# start_angles = [-2.9, 108.98, -133.94, 20.47, 5.18, -45.08]
# # 목표 엔드이펙터 좌표: [-12.9, -59.6, 251.5, -96.26, -44.51, -83.29]

# print("[INFO] 시작 좌표로 이동 중...")
# mc.send_angles(start_angles, speed, 0)
# time.sleep(1)


# coords = mc.get_coords()
# print(f"현재 엔드이펙터 좌표: {coords}")
# # 105.4, -65.2, 266.6, -85.23, -45.54, -92.7

# mc.set_gripper_value(100, 50)
# time.sleep(1)



start_coords1 = [100, -21.42, 337.29, -95.42, -44.74, -84.67]

print("[INFO] 시작 좌표로 이동 중...")
mc.send_coords(start_coords1, speed, 0)
time.sleep(1)

mc.set_gripper_value(100, 50)
time.sleep(1)

coords = mc.get_coords()
print(f"현재 엔드이펙터 좌표: {coords}")
# 115.0, -66.2, 256.6, -98.63, -44.49, -83.29