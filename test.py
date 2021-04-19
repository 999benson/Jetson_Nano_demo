import random
import numpy as np
import cv2

from EyeExam import EyeExam

IMGE_PATH = 'E1.png'
WINDOW_NAME = 'JETSON_NANO_DEMO'


def test():
    # env variables
    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(WINDOW_NAME, 640, 480)
    # cv2.moveWindow(WINDOW_NAME, 0, 0)
    # cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    leftEyeExam = EyeExam(IMGE_PATH)
    lResult = 0
    prev_gesture = []
    while True:

        #######################
        ### Detect Distance ###
        #######################

        imgE = leftEyeExam.GenerateQuestion()  # dist
        # cv2.imshow(WINDOW_NAME, imgE)

        ###########################
        ### Detect face and eye ###
        ###########################

        gesture = input()

        prev_gesture.append(gesture)
        prev_gesture = prev_gesture[-11:]

        if not all([x == None or x == '' for x in prev_gesture[:-1]]):
            print('skip this frame')
            continue


        if leftEyeExam.CheckAns(gesture):
            lResult = leftEyeExam.result
            break

        key = cv2.waitKey(1)
        if key == 27:  # space
            print('Halt!!!')
            return

    print(lResult)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
