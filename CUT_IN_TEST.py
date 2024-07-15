import time
import cv2
import numpy as np
import glob
import random

'''Video Used
https://drive.google.com/file/d/1j-2HLHtt-DwkFXXZUOf01p4AOFnWlDJ9/view?usp=drive_link
'''

net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
classes = ["car"]
images_path = glob.glob("cars_test1/*.jpg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
random.shuffle(images_path)
total = 0
count = 0
t1 = time.time()
for img_path in images_path:
    img = cv2.imread(img_path)
    if img is None:
        continue
    total += 1

    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # detection of objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.7)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
            if label == "car":
                count += 1

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

t2 = time.time()
cv2.destroyAllWindows()

print(f"{count} out of {total} detected\n")
print(f"Time Taken = {t2 - t1:.2f} seconds")
