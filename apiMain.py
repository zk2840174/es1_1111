from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel, UUID4
from typing import Union
from elasticsearch import Elasticsearch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

import uuid

# Elasticsearch 클라이언트 설정
es = Elasticsearch("http://localhost:9200", basic_auth=("elastic", "changeme"))

# CLIP 모델과 프로세서 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

app = FastAPI()


# 이미지 임베딩 생성 함수
def get_image_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    image_embeddings = model.get_image_features(**inputs)
    return image_embeddings.squeeze().tolist()  # JSON에 적합한 리스트 형태로 변환


# FastAPI에서 사용할 요청 모델 정의
class ImageEmbeddingRequest(BaseModel):
    image_id: str


@app.post("/insert-image-embedding/")
async def insert_image_embedding( file: UploadFile):
    """
    이미지 임베딩을 생성하고 Elasticsearch에 삽입하는 API 엔드포인트입니다.

    Args:
        request (ImageEmbeddingRequest): 이미지 ID를 포함하는 요청 데이터
        file (UploadFile): 업로드된 이미지 파일
    """
    try:
        # 이미지 파일 읽기
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        random_uuid = uuid.uuid4()
        # 임베딩 생성
        embedding = get_image_embedding(image)

        # Elasticsearch에 삽입할 문서 생성
        document = {
            "image_id": random_uuid,
            "image_embedding": embedding
        }

        # Elasticsearch에 문서 삽입
        es.index(index="image_embeddings", id=random_uuid, document=document)
        return {"message": f"Inserted embedding for image ID: {random_uuid}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-similar-images/")
async def search_similar_images(file: UploadFile, top_k: int = 5):
    """
    유사 이미지를 검색하는 API 엔드포인트입니다.

    Args:
        file (UploadFile): 업로드된 이미지 파일
        top_k (int): 반환할 유사 이미지의 개수 (기본값: 5)
    """
    try:
        # 검색할 이미지 임베딩 생성
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        query_embedding = get_image_embedding(image)

        # 유사 이미지 검색 쿼리 작성
        query = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'image_embedding') + 1.0",
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            }
        }

        # Elasticsearch에 쿼리 전송
        response = es.search(index="image_embeddings", body=query)

        # 검색 결과 반환
        similar_images = [
            {"image_id": hit["_source"]["image_id"], "score": hit["_score"]}
            for hit in response["hits"]["hits"]
        ]

        return {"similar_images": similar_images}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))