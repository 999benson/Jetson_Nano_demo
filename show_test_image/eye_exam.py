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

        while count < 2:
            t = ti.show_test_image(testingValue)
            cv2.imshow('Image', t[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print(t[1] + " " + str(testingValue))
            ans = input ("answer :")

            if ans == t[1]:
                testingValue += 0.1
                consecutive = False
            elif consecutive:
                testingValue -= 0.1
                consecutive = False
                if testingValue == 0:
                    count = 2
                else:
                    count += 1
            else:
                consecutive = True
        
        if status == "left":
            lResult = testingValue
            status = "right"
        else:
            rResult = testingValue
            status = "left"
    
    return lResult, rResult


r = eye_exam()
print(r)