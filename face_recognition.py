import dlib
import cv2
import os
import numpy as np
from imutils import face_utils
import argparse


#load dlib models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--faces", help="directory of known faces", default='./faces/')
ap.add_argument("-im", "--image", help="image to predict", default='./ex2.jpg')
args = vars(ap.parse_args())
                
                
img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)
for rect in rects:
    shape = sp(img, rect)
    fd = np.array(facerec.compute_face_descriptor(img, shape))
    
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #print(face_descriptor)
    
    name = 'UNKNOWN'
    cd = 1.0
    for i in os.listdir(args["faces"]):
        im = args["faces"] + i
        print(im)
        im = cv2.imread(im)
        gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        rec = detector(gr, 0)[0]
        sh = sp(im, rec)
        fd1 = np.array(facerec.compute_face_descriptor(im, sh))
        di = np.linalg.norm(fd - fd1)
        if di < 0.5 and di < cd:
            name = os.path.splitext(i)[0]
            cd = di
    cv2.putText(img, name, (x,abs(y-10)),0,0.01 * h,(255,255,255),2)
    print(name)
cv2.imshow('vid', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


