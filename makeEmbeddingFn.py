from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


# 모델과 프로세서를 전역 변수로 설정하여 매번 로드하지 않도록 합니다.
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_path):
    """
    주어진 이미지 경로의 임베딩 벡터를 반환하는 함수입니다.

    Args:
        image_path (str): 이미지 파일 경로

    Returns:
        torch.Tensor: 이미지 임베딩 벡터
    """
    # 이미지 불러오기
    image = Image.open(image_path)

    # 이미지 임베딩 생성
    inputs = processor(images=image, return_tensors="pt")
    image_embeddings = model.get_image_features(**inputs)

    return image_embeddings

# 함수 사용 예시
# image_path = "D:\\zzz\\IMG_1028.JPG"
# embedding = get_image_embedding(image_path)
# print("Image Embedding:", embedding)