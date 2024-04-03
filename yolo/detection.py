import cv2
from ultralytics import YOLO

model = YOLO('models/EPI-Detect.pt')


video = cv2.VideoCapture('videos/epi-1.mp4')
#video = cv2.VideoCapture(1,cv2.CAP_DSHOW)

while True:
    check, img = video.read()
    resultado = model.predict(img)

    for obj in resultado:
        names = obj.names

        for item in obj.boxes:
            x1,y1,x2,y2 = item.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            cls = int(item.cls[0])
            nameClass = names[cls]
            conf = round(float(item.conf[0]),2)
            text = f'{nameClass} - {conf}'
            cv2.putText(img,text,(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
            if nameClass == 'helmet' or nameClass == 'vest':
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),5)
            elif nameClass == 'no-helmet' or nameClass == 'no-vest':
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)


    cv2.imshow('IMG',img)
    if cv2.waitKey(1) == 27:
        break
