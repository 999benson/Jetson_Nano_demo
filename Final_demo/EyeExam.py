import cv2
import random

SCREEN_CONSTANT = 50
ALLOW_ANSWER = ['up', 'down', 'left', 'right', 'Shaking Hand']


class EyeExam(object):
    def __init__(self, path):
        self.dict = {0: 2, 0.1: 0}
        self.imgE = cv2.imread(path)
        self.result = 0.1
        self.ans = None

    def GenerateQuestion(self, distance: float = 6):
        # length for 1.0(or 20/20) is 0.9cm

        showSize = int(SCREEN_CONSTANT / self.result * distance / 6)
        # print(showSize)

        if self.ans == None:
            r = random.random()
            self.ans = "right"

            if r < 0.25:
                self.ans = "down"
            elif r < 0.5:
                self.ans = "up"
            elif r < 0.75:
                self.ans = "left"
            print('GenerateQuestion', self.ans)

        if self.ans == "down":
            img = cv2.rotate(self.imgE, cv2.ROTATE_90_CLOCKWISE)
        elif self.ans == "up":
            img = cv2.rotate(self.imgE, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.ans == "left":
            img = cv2.flip(self.imgE, -1)
        elif self.ans == 'right':
            img = self.imgE

        img = cv2.resize(img, (showSize, showSize),
                         interpolation=cv2.INTER_AREA)

        return img

    def CheckAns(self, resp):
        result = self.result
        dict = self.dict
        exam_end = False

        if self.ans == None or resp == None or resp not in ALLOW_ANSWER:
            return exam_end

        print('Check Answer   ', self.ans, '  ', resp)
        if resp == self.ans:
            if result in dict:
                dict[result] += 1
                if (result + 0.1) in dict:
                    if dict[result] >= 2 and dict[result + 0.1] <= -2:
                        exam_end = True
            else:
                dict[result] = 1
            result += 0.1
        else:
            if result in dict:
                dict[result] -= 1
                if (result - 0.1) in dict:
                    if dict[result - 0.1] >= 2 and dict[result] <= -2:
                        result -= 0.1
                        exam_end = True
            else:
                dict[result] = -1

            if result != 0.1:
                result -= 0.1

        self.ans = None
        self.result = max(result, 0.1)
        self.dict = dict
        print('After Check Answer    ', self.result)
        return exam_end
