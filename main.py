import random
import numpy as np
import cv2

from TSM import GestureDetector
from EyeExam import EyeExam

IMGE_PATH = 'E1.png'
WINDOW_NAME = 'JETSON_NANO_DEMO'


def main():
    print("Open camera...")
    cap = cv2.VideoCapture(0)

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    gd = GestureDetector()
    print("Ready!")
    while True:
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255

        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s') or key == 32:  # space
            print('Start!!')
            break

    leftEyeExam = EyeExam(IMGE_PATH)
    lResult = 0
    prev_gesture = []
    while True:
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255

        #######################
        ### Detect Distance ###
        #######################

        imgE = leftEyeExam.GenerateQuestion()  # dist
        cv2.imshow(WINDOW_NAME, imgE)

        ###########################
        ### Detect face and eye ###
        ###########################

        gesture, img = gd(img)
        prev_gesture.append(gesture)
        prev_gesture = prev_gesture[-10:]

        if not all([x == None for x in prev_gesture]):
            continue

        if leftEyeExam.CheckAns(gesture):
            lResult = leftEyeExam.result
            break

        key = cv2.waitKey(1)
        if key == 27:  # space
            print('Halt!!!')
            return

    print(lResult)

    cap.release()
    cv2.destroyAllWindows()

