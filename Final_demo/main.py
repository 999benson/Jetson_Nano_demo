import random
import numpy as np
import cv2
import random
import time
import imutils
from imutils.video import VideoStream
from imutils import face_utils
from TSM import GestureDetector
from EyeExam import EyeExam
from paho.mqtt import client as mqtt_client

broker = "172.20.10.2"
port = 1883
topic = "eye_exam/send"
topic2 = "eye_exam/ack"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 1000)}'
received_msg = "none"

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set("ntust", "123")
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def on_message(client, userdata, msg):
    global received_msg
    received_msg = msg.payload
    #print(msg.topic+" "+ msg.payload)

# dlib's built-in HOG detector, but less accurate), then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
#detector = dlib.get_frontal_face_detector()
#detector = cv2.CascadeClassifier('/home/ai/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#predictor = dlib.shape_predictor('shape_68.dat')
#(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
#(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

IMGE_PATH = 'E1.png'
WINDOW_NAME = 'JETSON_NANO_DEMO'


def main():
    global received_msg
    print("Open camera...")
    client = connect_mqtt()
    client.loop_start()
    #cap = cv2.VideoCapture(0)
    vs = VideoStream(src=0).start()

    # set a lower resolution for speed up
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # env variables
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 480)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)

    gd = GestureDetector()
    print("Ready!")
    while True:
       # _, img = cap.read()  # (480, 640, 3) 0 ~ 255
        img = vs.read()
        img = imutils.resize(img,width=300)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s') or key == 32:  # space
            print('Start!!')
            break

    msg = "right"
    temp = b'turn right'
    client.publish(topic, msg) ##eye_exam/send
    print(temp,'push done')
    client.subscribe(topic2) ## eye_exam/ack
    client.on_message = on_message
    while True:
        time.sleep(1.5)
        if received_msg == temp: ##RECEIVE CORRECTLY
            received_msg = ""
            break
        client.publish(topic, msg)##if error ,resent request
    print(temp,'received successs')

    leftEyeExam = EyeExam(IMGE_PATH)
    lResult = 0
    prev_gesture = []
    i_frame = -1 
    while True:
        i_frame += 1
        #_, img = cap.read()  # (480, 640, 3) 0 ~ 255
        img = vs.read()
        img = imutils.resize(img,width=300)
        #######################
        ### Detect Distance ###
        #######################

        imgE = leftEyeExam.GenerateQuestion()  # dist

        cv2.imshow(WINDOW_NAME, imgE)
        cv2.imshow('WINDOW_NAME', img)


        gesture, img = gd(img)
        prev_gesture.append(gesture)
        prev_gesture = prev_gesture[-11:]



        if not all([x == None for x in prev_gesture[:-1]]):
            continue

        if leftEyeExam.CheckAns(gesture):
            lResult = leftEyeExam.result
            break

        key = cv2.waitKey(1)
        if key == 27:  # space
            print('Halt!!!')
            return

    print('left eye vision',lResult)
    msg = "left"
    temp = b'turn left'
    client.publish(topic, msg) ##eye_exam/send
    print(temp,'push done')
    client.subscribe(topic2) ## eye_exam/ack
    client.on_message = on_message
    while True:
        time.sleep(1.5)
        if received_msg == temp: ##RECEIVE CORRECTLY
            received_msg = ""
            break
        client.publish(topic, msg)##if error ,resent request
    print(temp,'received successs')
    rightEyeExam = EyeExam(IMGE_PATH)
    lResult = 0
    prev_gesture = []
    i_frame = -1 
    while True:
        i_frame += 1
        #_, img = cap.read()  # (480, 640, 3) 0 ~ 255
        img = vs.read()
        img = imutils.resize(img,width=300)
        #######################
        ### Detect Distance ###
        #######################

        imgE = rightEyeExam.GenerateQuestion()  # dist

        cv2.imshow(WINDOW_NAME, imgE)
        cv2.imshow('WINDOW_NAME', img)


        gesture, img = gd(img)
        prev_gesture.append(gesture)
        prev_gesture = prev_gesture[-11:]



        if not all([x == None for x in prev_gesture[:-1]]):
            continue

        if rightEyeExam.CheckAns(gesture):
            rResult = rightEyeExam.result
            break

        key = cv2.waitKey(1)
        if key == 27:  # space
            print('Halt!!!')
            return

    print('Right Eye vision',rResult)
    #cap.release()
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == '__main__':
    main()
