import cv2
import random

def show_test_image(vision_size:float, distance:float = 6):
    # length for 1.0(or 20/20) is 0.9cm

    screen_constant = 50
    showSize = int(screen_constant / vision_size * distance / 6)
    # print(showSize)
    img = cv2.imread('C:/Users/user/Downloads/show_test_image/E1.png')

    r = random.random()
    ans = "right"

    if r < 0.25:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        ans = "down"
    elif r < 0.5:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ans = "up"
    elif r < 0.75:
        img = cv2.flip(img, -1)
        ans = "left"

    img = cv2.resize(img, (showSize, showSize), interpolation = cv2.INTER_LINEAR)
    return img, ans

# t = show_test_image(0.9)
# cv2.imshow('Image', t[0])
# print(t[1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for i in range(10):
#     img2 = img
#     r = random.random()
#     print(r)
    
#     if r < 0.25:
#         img2 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#     elif r < 0.5:
#         img2 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
#     elif r < 0.75:
#         img2 = cv2.flip(img, -1)

#     cv2.imshow('Image', img2)
#     cv2.waitKey(0)
