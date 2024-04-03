import cv2
from ultralytics import YOLO

model = YOLO('models/yolov8n-cls.pt')

img = cv2.imread('pic/images.jpg')

resultado = model.predict(img)

for obj in resultado:
    name = obj.names
    #obj.show()
    #print(obj.probs)
    top5 = obj.probs.top5
    for item in top5:
        print(name[item])

    obj.show()