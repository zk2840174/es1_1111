

from elasticsearch import Elasticsearch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Elasticsearch 클라이언트 설정
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "changeme")
)  # 본인의 Elasticsearch 주소와 포트를 사용하세요

# CLIP 모델과 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    image_embeddings = model.get_image_features(**inputs)
    return image_embeddings.squeeze().tolist()  # 텐서를 리스트 형태로 반환하여 JSON에 적합하게 만듦

def insert_image_embedding(image_id, image_path):
    """
    이미지 임베딩을 Elasticsearch에 삽입하는 함수입니다.

    Args:
        image_id (str): 이미지의 고유 ID
        image_path (str): 이미지 파일 경로
    """
    embedding = get_image_embedding(image_path)

    # Elasticsearch에 삽입할 문서 생성
    document = {
        "image_id": image_id,
        "image_embedding": embedding
    }

    # Elasticsearch에 문서 삽입
    es.index(index="image_embeddings", id=image_id, document=document)
    print(f"Inserted embedding for image ID: {image_id}")

# 함수 사용 예시
# image_path = "D:\\zzz\\IMG_1028.JPG"
# image_id = "unique_image_id"
# insert_image_embedding(image_id, image_path)


import os
from pathlib import Path

def list_files_in_folder_absolute(folder_path):
    return [os.path.abspath(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]



fileList = list_files_in_folder_absolute("D:\\zzz\\food")

for file in fileList:
    file_path = Path(file)
    file_name = file_path.name
    insert_image_embedding(file_name, file)