import cv2
import numpy as np
import time

net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

class_labels = []
with open('coco.names', 'r') as f:
    class_labels = f.read().splitlines()

lane_x1 = 414
lane_x2 = 490
lane_x3 = 644
lane_x4 = 177
lane_y1 = 383
lane_y2 = 530
lane_ymid = (lane_y1 + lane_y2) / 2

video_path = "https://drive.google.com/file/d/1-BvKSX6vGLZYOS8_OiLQXsHu1hsyC4qJ/view?usp=drive_link"
cap = cv2.VideoCapture(video_path)
def get_roi(x, y, img):
    slope1 = (lane_y2 - lane_y1) / (lane_x4 - lane_x1)
    slope2 = (lane_y2 - lane_y1) / (lane_x3 - lane_x2)
    if (y >= (slope1 * (x - lane_x4) + lane_y2) and y >= (slope2 * (x - lane_x3) + lane_y2) and y >= lane_ymid and y < lane_y2):
        return 2
    elif (y >= (slope1 * (x - lane_x4) + lane_y2) and y >= (slope2 * (x - lane_x3) + lane_y2) and y >= lane_y1 and y < lane_ymid):
        return 1
    else:
        return 0

frame_count = 0
while True:
    if frame_count % 10 != 0:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, None, fx=1, fy=1)
        frame_count += 1
        continue
    frame_count += 1

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=1, fy=1)
    frame_height, frame_width, frame_channels = frame.shape
    lane_points = np.array([[lane_x4, lane_y2], [lane_x1, lane_y1], [lane_x2, lane_y1], [lane_x3, lane_y2]])
    cv2.polylines(frame, [lane_points], True, (0, 255, 255), 2)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layer_names)
    boxes = []
    class_ids = []
    confidences = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width + 20)
                h = int(detection[3] * frame_height + 20)

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
                label = str(class_labels[class_ids[i]])
                confidence = str(round(confidences[i], 2))

                if get_roi(x + w / 2, y + h, frame) == 2:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Danger Zone", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                elif get_roi(x + w / 2, y + h, frame) == 1:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    cv2.putText(frame, "Warning Zone", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 165, 255), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Safe Zone", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(200) == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
