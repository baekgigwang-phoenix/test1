import time
import cv2
from pymycobot.mycobot280 import MyCobot280

# 로봇 연결
mc = MyCobot280('/dev/ttyJETCOBOT', 1000000)
print("로봇에 연결되었습니다.")
mc.thread_lock = True

# 지정된 좌표로 이동
mc.send_coords([191.9, -62.7, 246.9, -179.17, 0.25, -40.51], 50, 0)
time.sleep(1)

# 카메라 시작
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("▶ 실시간 스트리밍 중입니다.")
print("  - 'c': 모터 해제 (손으로 로봇 조작 가능)")
print("  - 'p': 현재 포즈 출력")
print("  - 'q': 종료")

# 모터 락 해제 여부
released = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    cv2.imshow('Camera View', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and not released:
        mc.release_all_servos()
        released = True
        print("✅ 모터 락 해제됨. 손으로 로봇을 조작하세요.")

    elif key == ord('p') and released:
        coords = mc.get_coords()
        angles = mc.get_angles()
        print(f"[포즈 출력] 좌표: {coords}")
        print(f"[포즈 출력] 각도: {angles}")

    elif key == ord('q'):
        print("종료합니다.")
        break

cap.release()
cv2.destroyAllWindows()

if released:
    print("🔒 모터를 다시 락(고정)합니다.")
    mc.focus_all_servos()

