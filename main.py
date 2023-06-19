import os
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import numpy as np
import cv2
import shutil
import yaml
import warnings
warnings.filterwarnings("ignore")

from ultralytics import YOLO
from glob import glob
from tqdm import tqdm
from IPython.display import clear_output
from sklearn.model_selection import train_test_split


SEED = 42
BATCH_SIZE = 4
MODEL = "v2"
EPOCH=300

with open("./classes.txt", "r") as reader:
    lines = reader.readlines()
    classes = [line.strip().split(",")[1] for line in lines]

yaml_data = {
              "names": classes,
              "nc": len(classes),
              "path": "/home/viplab/Dayoung/dacon/dataset/yolo/",
              "train": "train",
              "val": "valid",
              "test": "test",
              "hsv_h": 0.015,  # image HSV-Hue augmentation (fraction)
              "hsv_s": 0.7,  # image HSV-Saturation augmentation (fraction)
              "hsv_v": 0.4,  # image HSV-Value augmentation (fraction)
              "degrees": 0.5,  # image rotation (+/- deg)
              "translate": 0.1,  # image translation (+/- fraction)
              "scale": 0.2,  # image scale (+/- gain)
              "fliplr": 0.5, # image flip left-right (probability)
              "mosaic": 0.3,  # image mosaic (probability)
              "mixup": 0.1  # image mixup (probability)
            }

with open("./dataset/yolo/custom.yaml", "w") as writer:
    yaml.dump(yaml_data, writer)

model = YOLO("yolov8x")
print(type(model.names), len(model.names))

print(model.names)
results = model.train(
    data="./dataset/yolo/custom.yaml",
    imgsz=(1024, 1024),
    epochs=EPOCH,
    batch=BATCH_SIZE,
    patience=5,
    workers=16,
    device=0, 
    project=f"{MODEL}",
    seed=SEED,
    optimizer="AdamW",
    lr0=1e-3,
    hsv_h= 0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s= 0.7,  # image HSV-Saturation augmentation (fraction)
    hsv_v= 0.4,  # image HSV-Value augmentation (fraction)
    degrees= 0.5,  # image rotation (+/- deg)
    translate= 0.1,  # image translation (+/- fraction)
    scale= 0.2,  # image scale (+/- gain)
    fliplr= 0.5, # image flip left-right (probability)
    mosaic= 0.3,  # image mosaic (probability)
    mixup= 0.1  # image mixup (probability)
    )

def get_test_image_paths(test_image_paths):    
    for i in range(0, len(test_image_paths), BATCH_SIZE):
        yield test_image_paths[i:i+BATCH_SIZE]

model = YOLO("./v2/train/weights/best.pt")
test_image_paths = glob("./dataset/yolo/test/*.png")
print(len(test_image_paths))
for i, image in tqdm(enumerate(get_test_image_paths(test_image_paths)), total=int(len(test_image_paths)/BATCH_SIZE)):
    model.predict(image, imgsz=(1024, 1024), iou=0.2, conf=0.5, save_conf=True, save=False, save_txt=True, project=f"{MODEL}", name="predict",
                  exist_ok=True, device=0, augment=True, verbose=False)
    if i % 5 == 0:
        clear_output(wait=True)

def yolo_to_labelme(line, image_width, image_height, txt_file_name):    
    file_name = txt_file_name.split("/")[-1].replace(".txt", ".png")
    class_id, x, y, width, height, confidence = [float(temp) for temp in line.split()]
    
    x_min = int((x - width / 2) * image_width)
    x_max = int((x + width / 2) * image_width)
    y_min = int((y - height / 2) * image_height)
    y_max = int((y + height / 2) * image_height)
    
    return file_name, int(class_id), confidence, x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max

infer_txts = glob(f"{MODEL}/predict/labels/*.txt")

results = []
for infer_txt in tqdm(infer_txts):
    base_file_name = infer_txt.split("/")[-1].split(".")[0]
    base_file_name = base_file_name[7:]
    imgage_height, imgage_width = cv2.imread(f"./dataset/yolo/test/{base_file_name}.png").shape[:2]        
    with open(infer_txt, "r") as reader:        
        lines = reader.readlines()        
        for line in lines:
            results.append(yolo_to_labelme(line, imgage_width, imgage_height, infer_txt))

df_submission = pd.DataFrame(data=results, columns=["file_name", "class_id", "confidence", "point1_x", "point1_y", "point2_x", "point2_y", "point3_x", "point3_y", "point4_x", "point4_y"])
df_submission.to_csv("./results/yolov8x_adamw_ep"+str(EPOCH) + "_by"+str(BATCH_SIZE) + ".csv", index=False)
