import cv2
import mediapipe as mp


cap = cv2.VideoCapture(1)
mpHands = mp.solutions.hands
hands = mpHands.Hands()

while True : 
    ret , img = cap.read ()
    if ret : 
        imgRGB = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)
        print (result)
        
        cv2.imshow("img" , img)
    if cv2.waitKey(1) == ord ('q') :  
        break
