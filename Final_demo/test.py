import random
import numpy as np
import cv2

from EyeExam import EyeExam

IMGE_PATH = 'E3.png'
WINDOW_NAME = 'aiiuii'


def test():
    print("Open camera...")
    cap = cv2.VideoCapture(0)

    # set a lower resolution for speed up
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE |
                    cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    print("Ready!")
    t1 = 35
    t2 = 125
    i = 125
    j = 125
    while True:
        _, img = cap.read()  # (480, 640, 3) 0 ~ 255

        # convert the image to grayscale, blur it, and detect edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, t1, t2)
	#for c in contours:
	#		peri = cv2.arcLength(c, True)
	#		approx = cv2.approxPolyDP(c, 0.015 * peri, True)
	#		if len(approx) == 4:
	#			x,y,w,h = cv2.boundingRect(approx)
	#			cv2.rectangle(img,(x,y),(x+w,y+h),(36,255,12),2) 
        _, bin = cv2.threshold(gray, i, j, cv2.THRESH_BINARY_INV)
	#gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#blurred = cv2.GaussianBlur(gray2, (5, 5), 0)
	#ret, binary = cv2.threshold(blurred,220,255,cv2.THRESH_BINARY) 
        contours, hierarchy = cv2.findContours(bin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours: ##FIND WHITE BOARD
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.15 * peri, True)
                #cv2.drawContours(bin,[c],-1,(0,0,255),3)
                if len(approx) == 4:
                        x,y,w,h = cv2.boundingRect(approx)
                        cv2.drawContours(img,[c],-1,(0,0,255),3)
        contours, hierarchy = cv2.findContours(bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect_like_contours = [cnt for cnt in contours if len(cnt) == 4]
        #print(type(rect_like_contours))
        #print(len(rect_like_contours))
        if len(rect_like_contours) > 0:
            draw_img = cv2.drawContours(img.copy(), rect_like_contours, -1, (0, 0, 255), 3)
        else:
            draw_img = bin 

        cv2.imshow(WINDOW_NAME, bin)
        cv2.imshow('Contours',img)

        key = cv2.waitKey(1)
        if key == 27:  # space
            print('Halt!!')
            break
        if key & 0xff == ord('f'):  # space
            i +=1
        if key & 0xff == ord('d'):  # space
            i -=1
        if key & 0xff == ord('j'):  # space
            j +=1
        if key & 0xff == ord('k'):  # space
            j -=1

        print(i, j)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()
