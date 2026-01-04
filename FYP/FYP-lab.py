import cv2
import mediapipe as mp
import pyautogui
import time
import math

# ===============================
# 基础设置
# ===============================
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0  # 取消 pyautogui 自带延迟

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # ★ 降分辨率
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 60)

screen_w, screen_h = pyautogui.size()

# ===============================
# Mediapipe
# ===============================
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7  # ★ 关键
)
mpDraw = mp.solutions.drawing_utils

# ===============================
# 平滑参数
# ===============================
smooth = 0.18          # ★ 指数平滑系数（越小越稳）
dead_zone = 4          # ★ 忽略微小抖动（像素）

prev_x, prev_y = None, None

# 点击控制
click_delay = 0.35
last_left = 0
last_right = 0

# ===============================
# 主循环
# ===============================
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark

        # -------------------------------
        # 手指状态
        # -------------------------------
        index_up = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        thumb_up = lm[4].y < lm[3].y

        # -------------------------------
        # 鼠标移动（只在食指伸出）
        # -------------------------------
        if index_up and not middle_up:
            x = lm[8].x * screen_w
            y = lm[8].y * screen_h

            if prev_x is None:
                prev_x, prev_y = x, y

            dx = x - prev_x
            dy = y - prev_y

            # ★ 忽略抖动
            if abs(dx) > dead_zone or abs(dy) > dead_zone:
                new_x = prev_x + dx * smooth
                new_y = prev_y + dy * smooth

                pyautogui.moveTo(new_x, new_y, _pause=False)

                prev_x, prev_y = new_x, new_y

        now = time.time()

        # -------------------------------
        # 左键
        # -------------------------------
        if thumb_up and not index_up and not middle_up:
            if now - last_left > click_delay:
                pyautogui.click()
                last_left = now

        # -------------------------------
        # 右键
        # -------------------------------
        if middle_up and not index_up and not thumb_up:
            if now - last_right > click_delay:
                pyautogui.rightClick()
                last_right = now

        mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Hand Mouse (Smooth)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
