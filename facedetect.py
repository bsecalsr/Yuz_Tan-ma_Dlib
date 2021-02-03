import numpy as np
import cv2
import dlib
import pickle
import imutils


face_detector = cv2.CascadeClassifier ('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks (1).dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
FACE_DESC, FACE_NAME = pickle.load(open('trainset.pk', 'rb'))
video_capture = cv2.VideoCapture(0)

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
            for face_desc in FACE_DESC:
                d.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc0)))
            d = np.array(d)
            idx = d.argmin()
            if (idx).size > 0 :
              name = FACE_NAME[idx]
              print(name)
              frame = imutils.resize(frame, width=600)
              #frame = np.zeros((500,500,3),np.uint8)
              #cv2.putText(frame,'test',(100,10), cv2.FONT_HERSHEY_COMPLEX, 1.0,(255,255,255),lineType=cv2.LINE_AA)
              cv2.putText(frame, name, (x+10, y-50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0),4 )
              cv2.rectangle(frame,(x,y),  (x + w, y + h), (0,255,0), 2)
     cv2.imshow ('frame',frame)
     cv2.waitKey(1)
    