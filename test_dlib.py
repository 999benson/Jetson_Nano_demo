from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# dlib's built-in HOG detector, but less accurate), then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')


# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)
img = vs.read()

while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	print(gray,'=================')
	rects = detector(gray, 1) ##grayscale frame per frame
	

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# construct a dlib rectangle object from the Haar cascade
		# bounding box
		#rect = dlib.rectangle(int(x), int(y), int(x + w),
		#	int(y + h))
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		print(rect.top(),rect.bottom(),rect)
		img = frame[rect.top():rect.bottom(),rect.left():rect.right()]
		#cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 4, cv2.LINE_AA)
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		#print(shape)
		cv2.imshow("img", img) 

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		
		


	# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
		#cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
			#cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame

	#img = frame.copy()
	#gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#ret, binary = cv2.threshold(gray2,127,255,cv2.THRESH_BINARY) 
	#contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
	#cv2.drawContours(img,contours,-1,(0,0,255),3)   
	#cv2.imshow("img", img) 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	
# do a bit of cleanup
 
cv2.destroyAllWindows()
vs.stop()


