# results = []
# result = (0.1, 0.1)
# results.append(result)
# print(results[0])

import cv2
import random
import show_test_image as ti

status = "left"

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
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print(t[1] + " " + str(testingValue))
            print(r)
            ans = input ("answer: ")

            if ans == t[1]:
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
                    
        if status == "left":
            lResult = testingValue
            status = "right"
        else:
            rResult = testingValue
            status = "left"
    
    return lResult, rResult

result = eye_exam()
print(result)