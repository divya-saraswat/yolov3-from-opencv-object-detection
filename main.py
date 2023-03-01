import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

import util


# define constants
model_cfg_path = os.path.join('.', 'model', 'cfg', 'yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'yolov3.weights')
class_names_path = os.path.join('.', 'model', 'classes.names')

input_dir = "data/final_images"
output_dir = "results"
for img_name in os.listdir(os.path.join(".", input_dir)):
    print("_____________________________________________________________")
    print("Image Name: ", img_name)
    img_path = os.path.join(".", input_dir, img_name)

    # img_path = './pexels-diana-huggins-615369.jpg'

    # load class names
    with open(class_names_path, 'r') as f:
        class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
        f.close()

    # load model
    net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

    # load image

    img = cv2.imread(img_path)
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    img = cv2.edgePreservingFilter(img, flags=1, sigma_s=60, sigma_r=0.4)

    H, W, _ = img.shape

    # convert image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # get detections
    net.setInput(blob)

    detections = util.get_outputs(net)

    # bboxes, class_ids, confidences
    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        # [x1, x2, x3, x4, x5, x6, ..., x85]
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # apply nms
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    # plot
    reader = easyocr.Reader(['en'])

    for bbox_, bbox in enumerate(bboxes):
        xc, yc, w, h = bbox

        # cv2.putText(img,
        #             class_names[class_ids[bbox_]],
        #             (int(xc - (w / 2)), int(yc + (h / 2) - 20)),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             7,
        #             (0, 255, 0),
        #             15)
        license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :]
        img = cv2.rectangle(img,
                            (int(xc - (w / 2)), int(yc - (h / 2))),
                            (int(xc + (w / 2)), int(yc + (h / 2))),
                            (0, 255, 0),
                            10)
        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        _, license_plate_threshold = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
        output = reader.readtext(license_plate_threshold)
        print(output)
        print("________________________________________")
        for out in output:
            text_bbox, text, text_score = out
            print(text, text_score)

    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.savefig(os.path.join(".", output_dir, img_name.split(".")[0]) + "_detected" + ".jpg")

    plt.figure()
    # plt.imshow(cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB))
    plt.savefig(os.path.join(".", output_dir, "cropped_images", img_name.split(".")[0]) + "_detected_plate" + ".jpg")
    plt.show()
