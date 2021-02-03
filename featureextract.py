import numpy as np
import cv2
import os.path
import dlib
import pickle
import imutils

path = './facedata/'
face_detector = cv2.CascadeClassifier ('haarcascade_frontalface_default.xml')
#detector = dlib.get_frontal_face_detector()
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks (1).dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
FACE_DESC = []
FACE_NAME = []
FACE_DESC, FACE_NAME = pickle.load(open('trainset.pk', 'rb'))
video_capture = cv2.VideoCapture(0)

for fn in os.listdir(path):
    if fn.endswith('.jpg'):
        img = cv2.imread(path + fn) [:,:,::-1]
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_desc = model.compute_face_descriptor(img, shape, 10)
            FACE_DESC.append(face_desc)
            print('loading...',fn)
            FACE_NAME.append(fn[:fn.index('_')])
pickle.dump((FACE_DESC, FACE_NAME), open('trainset.pk','wb'))    

while True:
     _,frame = video_capture.read()
     #frame = str(np.array(frame, dtype=np.uint8))
     gray = cv2.cvtColor(np.array(frame, dtype=np.uint8), cv2.COLOR_BGR2GRAY)
     #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     faces = face_detector.detectMultiScale( gray, 1.3, 5)
     #faces = face_detector.detectMultiScale( gray)
     for (x, y, w, h) in faces:
        img = frame[y-10:y+h+10, x-10:x+w+10][:,:,::-1]
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_desc0 = model.compute_face_descriptor(img, shape, 1)
            d = []
            
     #cv2.imshow ('frame',frame)
     if cv2.waitKey(1) & 0xFF == ord('q'):
        break
     
     
        
