import cv2
import numpy as np
import time
net = cv2.dnn.readNet('yolov3.weights', 'yolo3.cfg')

classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()
x1, x2, x3, x4 = 415, 490, 644, 177
y1, y2 = 383, 530
ym = (y1 + y2) / 2

tpo, tpr = 0, 0
cap = cv2.VideoCapture("https://drive.google.com/file/d/1-BvKSX6vGLZYOS8_OiLQXsHu1hsyC4qJ/view?usp=drive_link")

def roi(x, y, img):
    m1 = (y2 - y1) / (x4 - x1)
    m2 = (y2 - y1) / (x3 - x2)
    if (y >= (m1 * (x - x4) + y2) and y >= (m2 * (x - x3) + y2) and y >= ym and y < y2):
        return 2
    elif (y >= (m1 * (x - x4) + y2) and y >= (m2 * (x - x3) + y2) and y >= y1 and y < ym):
        return 1
    else:
        return 0

c = 0
while True:
    if c % 10 != 0:
        _, img = cap.read()
        img = cv2.resize(img, None, fx=1, fy=1)
        height, width, channels = img.shape
        c += 1
        continue

    c += 1
    _, img = cap.read()
    if not _:
        break
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape
    pts = np.array([[x4, y2], [x1, y1], [x2, y1], [x3, y2]])
    cv2.polylines(img, [pts], True, (0, 255, 255), 2)
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    boxes, class_ids, confidences = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            if w * h < 200000:
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))

                if roi(x + w / 2, y + h, img) == 2:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, "Danger Zone - Careful!", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                elif roi(x + w / 2, y + h, img) == 1:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(img, "Warning Zone", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 165, 255), 2)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, "Safe Zone", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
