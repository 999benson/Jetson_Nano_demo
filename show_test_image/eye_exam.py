# results = []
# result = (0.1, 0.1)
# results.append(result)
# print(results[0])

import cv2
import random
import time
import show_test_image as ti

status = "left"

def detect_gestures():
    r = random.random()
    ans = "right"

    if r < 0.25:
        ans = "down"
    elif r < 0.5:
        ans = "up"
    elif r < 0.75:
        ans = "left"
    
    return ans

def eye_exam():
    global status
    rResult = 0.1
    lResult = 0.1
    print("Start eye exam!")

    for i in range(2):
        testingValue = 0.1
        count = 0
        consecutive = False
        r = {0: 2, 0.1: 0}

        while True:
            t = ti.show_test_image(testingValue)
            cv2.imshow('Image', t[0])
            cv2.waitKey(1)
            # cv2.destroyAllWindows()

            if not detect_face():
                continue

            if detect_eye() != status:
                continue

            print(t[1] + " " + str(testingValue))
            print(r)
            # ans = input ("answer: ")
            ans = detect_gestures()
            time.sleep(3)

            if ans == None:
                continue
            elif ans == t[1]:
                # print(testingValue in r)
                if testingValue in r:
                    r[testingValue] += 1
                    if (testingValue + 0.1) in r:
                        if r[testingValue] >= 2 and r[testingValue + 0.1] <= -2:
                            break
                else:
                    r[testingValue] = 1
                testingValue += 0.1
            else :
                if testingValue in r:
                    r[testingValue] -= 1
                    if (testingValue - 0.1) in r:
                        if r[testingValue - 0.1] >= 2 and r[testingValue] <= -2:
                            testingValue -= 0.1
                            break
                else:
                    r[testingValue] = -1
                
                if testingValue != 0.1:
                    testingValue -= 0.1
            
            while detect_gestures() != None:
                time.sleep(0)

            
            
                    
        if status == "left":
            lResult = testingValue
            status = "right"
        else:
            rResult = testingValue
            status = "left"
    
    return lResult, rResult

if __name__ == "__main__":
    result = eye_exam()
    print(result)