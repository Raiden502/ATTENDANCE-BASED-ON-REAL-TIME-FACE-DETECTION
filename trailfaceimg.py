# import libraries
import cv2
import face_recognition
import imutils
import time
import pickle
import datetime
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream
import dlib

dlib.DLIB_USE_CUDA = True


print("Loading Face Detector...")
protoPath = "face_detection_model/deploy.prototxt"
modelPath = "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("Loading Face Recognizer...")
recognizer = pickle.loads(open("output/recognizerE.pickle", "rb").read())
print("Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
p = list()
fps = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            vec =face_recognition.face_encodings(face,num_jitters=1,model="small")
            for i in zip(vec):
                preds = recognizer.predict(i)
                print(preds[0],datetime.datetime.now())
                p.append(preds[0])
                text = "{}".format(preds[0])
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    fps.update()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
fps.stop()
print(set(p))
cv2.destroyAllWindows()
vs.stop()
