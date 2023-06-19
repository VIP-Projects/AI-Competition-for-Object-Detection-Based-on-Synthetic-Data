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

if os.path.exists("./dataset/yolo"):
    shutil.rmtree("./dataset/yolo")

if not os.path.exists("./dataset/yolo/train"):
    os.makedirs("./dataset/yolo/train")
    
if not os.path.exists("./dataset/yolo/valid"):
    os.makedirs("./dataset/yolo/valid")
    
if not os.path.exists("./dataset/yolo/test"):
    os.makedirs("./dataset/yolo/test")    
    
if not os.path.exists("./results"):
    os.makedirs("./results")
    
def make_yolo_dataset(image_paths, txt_paths, type="train"):
    for image_path, txt_path in tqdm(zip(image_paths, txt_paths if not type == "test" else image_paths), total=len(image_paths)):
        source_image = cv2.imread(image_path, cv2.IMREAD_COLOR)        
        image_height, image_width, _ = source_image.shape
        
        target_image_path = f"./dataset/yolo/{type}/{os.path.basename(image_path)}"
        cv2.imwrite(target_image_path, source_image)
        
        if type == "test":
            continue
        
        with open(txt_path, "r") as reader:
            yolo_labels = []
            for line in reader.readlines():
                line = list(map(float, line.strip().split(" ")))
                class_name = int(line[0])
                x_min, y_min = float(min(line[5], line[7])), float(min(line[6], line[8]))
                x_max, y_max = float(max(line[1], line[3])), float(max(line[2], line[4]))
                x, y = float(((x_min + x_max) / 2) / image_width), float(((y_min + y_max) / 2) / image_height)
                w, h = abs(x_max - x_min) / image_width, abs(y_max - y_min) / image_height
                yolo_labels.append(f"{class_name} {x} {y} {w} {h}")
            
        target_label_txt = f"./dataset/yolo/{type}/{os.path.basename(txt_path)}"      
        with open(target_label_txt, "w") as writer:
            for yolo_label in yolo_labels:
                writer.write(f"{yolo_label}\n")
                

image_paths = sorted(glob("./dataset/train/*.png"))
txt_paths = sorted(glob("./dataset/train/*.txt"))

train_images_paths, valid_images_paths, train_txt_paths, valid_txt_paths = train_test_split(image_paths, txt_paths, test_size=0.1, random_state=SEED)

make_yolo_dataset(train_images_paths, train_txt_paths, "train")
make_yolo_dataset(valid_images_paths, valid_txt_paths, "valid")
make_yolo_dataset(sorted(glob("../dataset/test/*.png")), None, "test")
