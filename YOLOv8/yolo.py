from ultralytics import YOLO
import numpy as np
import cv2
import json

from pathlib import Path
from tqdm import tqdm

def yolo_model(weight_path):
    return YOLO(weight_path)  # load a custom model
    
def yolo_predict(model, data_path, agnostic_nms=False, conf=0.25, save=True):
    results = model(data_path, save = save, show_labels=False, show_conf= True, agnostic_nms= agnostic_nms, conf=conf)  # multi-class predict on an image
    bboxes = results[0].boxes
    
    print("object number : ", bboxes.xywhn.shape[0])

    return np.concatenate([bboxes.xywhn.cpu(), bboxes.conf.reshape(-1, 1).cpu()], axis=-1), 
# task="detect", source="/mnt/datasets/rebar_bale_detection/images/val", max_det=1000, conf=0.55, show_labels=False, show_conf=False, save=True, device="0", augment=False)  # predict on a Image

def yolo_crop_image(img, yolo_label):
    list_crop_image = []
    img_shape = img.shape
    for box in yolo_label:
        x, y, w, h, class_label = box
        x = x * img_shape[1]
        y = y * img_shape[0]
        w = w * img_shape[1]
        h = h * img_shape[0]
        
        # 타이트한 라벨링의 경우
        w += 30
        h += 30

        x = x - w//2
        y = y - h//2
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)


        # continue if the box is out of image
        if x < 0 or y < 0 or x+w > img_shape[1] or y+h > img_shape[0]:
            continue
        crop_img = img[max(y-10,0):min(y+h+10, img_shape[1]), max(x-10,0):min(x+w+10, img_shape[0])]
        w, h = crop_img.shape[1], crop_img.shape[0]
        if w > 30 and h > 30:
            x, y = max(x-10,0), max(y-10,0)
        
            list_crop_image.append((x, y, w, h, crop_img))
    # print("list_crop_image : ", len(list_crop_image))
    # for i in range(len(list_crop_image)):
    #     x, y, w, h, crop_img = list_crop_image[i]
    #     print("x, y, w, h : ", x, y, w, h)
    return list_crop_image

def save_predictions(processed_path, default_dir, project_id, uploaded_path, yolo_model0, conf):
    
    annotation = {"data": {"image" : default_dir + uploaded_path}, "project_id": project_id}
    
    list_boxes = yolo_predict(yolo_model0, processed_path, conf=conf, save=False)
    img = cv2.imread(processed_path)
    img_size = img.shape[:2]
    print(annotation)
    predictions = [{
        "model_version": "YOLOv8m",
        "score" : conf,
        "result" : []
    }]

    for idx, box in enumerate(list_boxes[0]):
        x, y, w, h, conf = box
        x = x - w/2
        y = y - h/2
        x *= 100
        y *= 100
        w *= 100
        h *= 100
        
        result = {
            "id": "result" + str(idx+1),
            "type": "rectanglelabels",
            "from_name": "label", "to_name": "image",
            "original_width": img_size[1], "original_height": img_size[0],
            "image_rotation": 0,
            "value": {
                "rotation": 0,
                "x": x, "y": y, "width": w, "height": h,
                "rectanglelabels": ["target"]       # target label 기입
            }
        }
        predictions[0]["result"].append(result)
        
    annotation["predictions"] = predictions
    print(len(predictions[0]["result"]))
    
    json_name = Path(processed_path).parent / (Path(processed_path).stem + ".json")
    
    json.dump(annotation, open(json_name, "w"), indent=4)

