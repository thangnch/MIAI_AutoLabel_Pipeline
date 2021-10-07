import os
import cv2
import numpy as np


# Ham tra ve output layer
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# Ham ve cac hinh chu nhat va ten class
def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = (255,0,0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


data_path  = "data"
class_name_file = "class_name.txt"
weight_file = "yolov4.weights"
config_file = "yolov4.cfg"

scale = 0.00392
conf_threshold = 0.5
nms_threshold = 0.4


# Doc ten cac class
classes = None
with open(class_name_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(weight_file, config_file)

# Bat dau doc tu thư mục
for file in os.listdir(data_path):

    if file[-3:] != "jpg":
        continue
    # Doc frame
    image = cv2.imread(os.path.join(data_path, file))

    # Resize va dua khung hinh vao mang predict
    image_width = image.shape[1]
    image_height = image.shape[0]

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    # Loc cac object trong khung hinh
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * image_width)
                center_y = int(detection[1] * image_height)
                w = int(detection[2] * image_width)
                h = int(detection[3] * image_height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Ve cac khung chu nhat quanh doi tuong
    label_file = os.path.join(data_path, file[:-3] + "txt")
    with open(label_file, "w") as f:
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], round(x), round(y), round(x + w), round(y + h))

            # Ghi vào file nhãn
            f.write("{} {} {} {} {}\n".format(0, x / image_width, y/image_height, w/image_width, h/image_height))



    cv2.imshow("object detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()