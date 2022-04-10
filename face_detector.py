import numpy as np
import numba
import os
import cv2
import time
import imutils
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
def detect_face(frame,faceNet,maskNet):
    h,w=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(224,224),(104.0,177.0,123.0))
    faceNet.setInput(blob)
    detections=faceNet.forward()
    print(detections.shape)
    faces=[]
    locs=[]
    preds=[]
    for i in range (0,detections.shape[2]):
        conf=detections[0,0,i,2]
        if(conf>0.5):
            box=detections[0,0,i,3:7] * np.array([w,h,w,h])
            X_start,Y_start,X_end,Y_end=box.astype("int")
            
            X_start,Y_start=(max(0,X_start),max(0,Y_start))
            X_end,Y_end=(min(w-1,X_end),min(h-1,Y_end))
            
            face=frame[Y_start:Y_end,X_start:X_end]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face=preprocess_input(face)
            faces.append(face)
            locs.append((X_start,Y_start,X_end,Y_end))
        if len(faces)>0 :
            faces=np.array(faces,dtype="float64")
            preds=maskNet.predict(faces,batch_size=32)
    return locs,preds
protopath=r"D:/IIITG/Projects/MaskDectection/face_detector/deploy.prototxt"
weightpath=r"D:/IIITG/Projects/MaskDectection/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet=cv2.dnn.readNet(protopath,weightpath)
maskNet=load_model("D:/IIITG/Projects/MaskDectection/mask_detector_model")
print("Video streaming start")
vs=VideoStream(src=0).start()
while True:
    frame=vs.read()
    frame=imutils.resize(frame,width=400)
    locs,preds=detect_face(frame,faceNet,maskNet)
    for box,pred in zip(locs,preds):
        X_start,Y_start,X_end,Y_end=box
        mask,withoutMask=pred
        labels="Mask" if mask>withoutMask else "NO Mask! You idiot, Ronald Weasley"
        color=(0,255,0) if labels =="Mask" else (0,0,255)
        labels="{}:{:.2f}%".format(labels,max(mask,withoutMask)*100)
        
        cv2.putText(frame,labels,(X_start,Y_start-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
        
        cv2.rectangle(frame,(X_start,Y_start),(X_end,Y_end),color,2)
    cv2.imshow("Fraame",frame)
    key=cv2.waitKey(1)* 0xFF
    
    if key==ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
