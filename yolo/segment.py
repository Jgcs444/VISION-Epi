import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('models/yolov8x-seg.pt')

video = cv2.VideoCapture('videos/epi-2.mp4')

while True:
    check, img = video.read()
    resultado = model.predict(img)

    for obj in resultado:
        for mask in obj.masks.data:
            maskConv = mask.cpu().numpy()
            maskConv = cv2.resize(maskConv,(img.shape[1],img.shape[0]))

            contours,_ = cv2.findContours(maskConv.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img,contours,-1,(0,255,9),3)

    cv2.imshow('IMG',img)
    if cv2.waitKey(1) ==27:
        break