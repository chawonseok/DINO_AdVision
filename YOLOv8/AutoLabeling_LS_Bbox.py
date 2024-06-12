from pathlib import Path

from utils.utils import load_src_dir, calculate_rgb_averages, image_processing
from yolo import yolo_predict, yolo_crop_image, yolo_model, save_crops, save_predictions

import torch.nn.functional as F
from time import time

import requests, os, json



# 1. src directory load

#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
src_dir = r""     # auto_test dataset
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####

image_paths = load_src_dir(src_dir)
print("image example: ", image_paths[0])
print("-"*100)

# # base rgb의 +- 30% 범위 내에 없을 경우 +- 30% 범위 내로 전체 이미지 조정(전처리 부분)
# # 0. 데이터셋 평균
# base_rgb = (39.272, 38.367, 45.294)

# # 2. image 전처리 및 모델 load
# processed_dir = Path(src_dir) / "processed"
# processed_dir.mkdir(parents=True, exist_ok=True)

# r_avg, g_avg, b_avg = calculate_rgb_averages(image_paths)
# print(f"RGB 채널별 평균값: R = {r_avg}, G = {g_avg}, B = {b_avg}")
# print("-"*100)

# # base rgb의 +- 30% 범위 내에 없을 경우 +- 30% 범위 내로 전체 이미지 조정

# image_processing(image_paths, base_rgb, r_avg, g_avg, b_avg, processed_dir)
# print("-"*100)
# processed_paths = load_src_dir(processed_dir)
# r_avg, g_avg, b_avg = calculate_rgb_averages(processed_paths)
# print(f"전처리된 RGB 채널별 평균값: R = {r_avg}, G = {g_avg}, B = {b_avg}")

processed_paths = image_paths

# 3. YOLO 모델 predict
yolo_dir = "yolov8m.pt"
yolo_model0 = yolo_model(yolo_dir)
conf = 0.25

yolo_start = time()
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
image_dir = "ObjectDetection/PlantSeedQuality/Onion/images" # label studio 안에 폴더
project_id = 70
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####
#### 베리 중요 이건 꼭 살펴볼 것 ####

default_dir = f"/data/local-files/?d={image_dir}/"
for processed_path in processed_paths:
    uploaded_path = Path(processed_path).name
    save_predictions(processed_path, default_dir, project_id, uploaded_path, yolo_model0, conf)

yolo_end = time()
print("yolo time : ", yolo_end - yolo_start)

# # json 파일을 Label Studio API로 업로드 (미구현)

# api_url = 
# api_key = 
# headers = {
#     "Authorization": f"Token {api_key}"
# }

# # JSON 파일이 있는 디렉토리 경로
# directory = processed_dir

# # 디렉토리 내 모든 파일에 대해 반복 수행
# for filename in os.listdir(directory):
#     if filename.endswith(".json"):
#         file_path = os.path.join(directory, filename)
        
#         # JSON 파일 읽기
#         with open(file_path, 'r', encoding='utf-8') as file:
#             data = json.load(file)
        
#             response = requests.post(api_url, headers=headers, files=data)
        
#         # 응답 확인
#         if response.status_code == 200:
#             print(f"Success: {filename}")
#         else:
#             print(f"Failed: {filename} with status code {response.status_code} - {response.text}")
        

    # client = LabelStudio(
    #     api_key=api_key,
    # )
    # client.tasks.create(
    #     data={"image": "https://example.com/image.jpg", "text": "Hello, world!"},
    #     project=1,
    # )
