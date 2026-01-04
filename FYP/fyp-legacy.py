import cv2
import mediapipe as mp
import pyautogui
import time

# -------------------------------
# 初始化摄像头
# -------------------------------
cap = cv2.VideoCapture(0)  # 可根据摄像头调整 0 或 1

# Mediapipe 手部模型
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# 获取屏幕尺寸
screen_w, screen_h = pyautogui.size()

# 平滑参数
smoothening = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# 控制点击频率，防止连点
click_delay = 0.3  # 秒
last_left_click = 0
last_right_click = 0

# -------------------------------
# 主循环
# -------------------------------
while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)  # 镜像翻转
    h, w, _ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmarks = handLms.landmark

            # -------------------------------
            # 手指状态
            # -------------------------------
            index_up = landmarks[8].y < landmarks[6].y   # 食指伸出
            middle_up = landmarks[12].y < landmarks[10].y  # 中指伸出
            thumb_up = landmarks[4].y < landmarks[3].y    # 大拇指伸出

            # -------------------------------
            # 鼠标移动
            # -------------------------------
            if index_up:
                x = landmarks[8].x * screen_w
                y = landmarks[8].y * screen_h

                # 平滑处理
                clocX = plocX + (x - plocX) / smoothening
                clocY = plocY + (y - plocY) / smoothening
                pyautogui.moveTo(clocX, clocY)
                plocX, plocY = clocX, clocY

            # -------------------------------
            # 左键点击：大拇指伸出且食指收回
            # -------------------------------
            current_time = time.time()
            if thumb_up and not index_up and not middle_up:
                if current_time - last_left_click > click_delay:
                    pyautogui.click()
                    last_left_click = current_time
                    cv2.putText(img, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # -------------------------------
            # 右键点击：中指伸出且食指收回
            # -------------------------------
            if middle_up and not index_up and not thumb_up:
                if current_time - last_right_click > click_delay:
                    pyautogui.rightClick()
                    last_right_click = current_time
                    cv2.putText(img, "Right Click", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # -------------------------------
            # 绘制骨架
            # -------------------------------
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Hand Control Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
