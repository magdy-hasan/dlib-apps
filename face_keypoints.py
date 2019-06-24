import dlib
import cv2
import os
import numpy as np
from imutils import face_utils
import argparse


#load dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords


ap = argparse.ArgumentParser()
ap.add_argument("-im", "--image", help="image to predict", default='./ex2.jpg')
args = vars(ap.parse_args())
                
                
img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)

for rect in rects:
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    shape = sp(img, rect)
    shape = shape_to_np(shape)
    # Draw the face landmarks on the screen.
    for (x, y) in shape:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    
cv2.imshow('vid', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


