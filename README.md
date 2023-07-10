# DACON: 합성데이터 기반 객체 탐지 AI 경진대회

<img alt="Html" src ="https://img.shields.io/badge/dacon Final rank-Top 4%25-lightblue?style=for-the-badge"/>

#### 합성 데이터를 활용한 자동차 탐지 AI 모델 개발 (23.05.08  - 23.07.07) - 김준용, 길다영
##### 📊 [PUBLIC] 29/859 (상위 4%) 점수: 0.95437
##### 📊 [PRIVATE] 27/859 (상위 4%) 점수: 0.95203

<br><br>

### File 설명

- <b>dataset.py</b>: 주어진 dataset을 yolo에 쓰일 dataset으로 변환하는 Code.

- <b>main.py</b>: yolo 이용해 detection과 evaluation하는 Code.



<br>

### Yolov8 사용
- Baseline에서 제안된 Faster-RCNN 모델이 아닌 코드 공유 커뮤니티에 제시된 Yolov8로 detection. <br>
  - *코드 공유: [yolo8 코드입니다](https://dacon.io/competitions/official/236107/codeshare/8414?page=1&dtype=recent)*
  - *코드 공유: [YOLOv8 커스텀 데이터 학습](https://github.com/neowizard2018/neowizard/blob/master/DeepLearningProject/YOLOv8_Object_Detection_Roboflow_Aquarium_Data.ipynb)*

- Our experiments에서는 다음과 같이 parameter를 setting할 때 가장 성능이 좋았음.

```
MODEL = "yolov8x"
BATCH_SIZE = 4,
EPOCH = 300,
OPTIMIZER = "Adamw",
IMGSZ = (1024,1024)
```

<br>

### 불균형 Class 해소 위해 Yolo에서 Data Augmentation 진행
- yolov8 official page를 참고하여 data augmentation 진행함.

  - *Ultralytics: [Configuration](https://docs.ultralytics.com/usage/cfg/#export)*

```
hsv_h= 0.015,  # image HSV-Hue augmentation (fraction)
hsv_s= 0.7,  # image HSV-Saturation augmentation (fraction)
hsv_v= 0.4,  # image HSV-Value augmentation (fraction)
degrees= 0.5,  # image rotation (+/- deg)
translate= 0.1,  # image translation (+/- fraction)
scale= 0.2,  # image scale (+/- gain)
fliplr= 0.5, # image flip left-right (probability)
mosaic= 0.3,  # image mosaic (probability)
mixup= 0.1  # image mixup (probability)
```

<br>


### 아쉬운 점
- 대회가 끝난 뒤, Detection에서도 Model Ensemble을 할 수 있는 코드 발견함.<br>
  - MAILAB-Yonsei: *[capsule_endoscopy_detection](https://github.com/MAILAB-Yonsei/capsule_endoscopy_detection)*

- Data Augmentation한 model들을 Ensemble 한다면, 더 좋은 성능 보일 거라 예상.


<br><br>


<b>출처 |</b> [DACON - 합성데이터 기반 객체 탐지 AI 경진대회](https://dacon.io/competitions/official/236107/overview/description) <br>
