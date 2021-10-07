import os
import cv2
import numpy as np

# Ham ve cac hinh chu nhat va ten class
def draw_prediction(img, x, y, x_plus_w, y_plus_h):
    color = (255,0,0)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, "--label-it--", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


data_path  = "data"

class_idx = {
"S": 0, "L":1,"B":2
}

# Bat dau doc tu folder
for file in os.listdir(data_path):

    if file[-3:] != "jpg":
        continue

    print("File anh hien tai: ", file)

    image = cv2.imread(os.path.join(data_path, file))
    image_width = image.shape[1]
    image_height = image.shape[0]

    label_file = os.path.join(data_path, file[:-3] + "txt")

    boxes = None
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            boxes = f.readlines()

    if len(boxes)>0:
        with open(label_file, "w") as f:
            for box in boxes:
                box = box.replace("\n","").split()
                print(box)

                x = float(box[1]) * image_width
                y = float(box[2]) * image_height
                w = float(box[3]) * image_width
                h = float(box[4]) * image_height
                draw_prediction(image, round(x), round(y), round(x + w), round(y + h))

                cv2.imshow("object detection", image)
                cv2.waitKey(1)
                re_class = input("Class: ")
                if re_class == '!':
                    break
                else:
                    f.write("{} {} {} {} {}\n".format(class_idx[re_class], box[1], box[2], box[3], box[4]))



cv2.destroyAllWindows()