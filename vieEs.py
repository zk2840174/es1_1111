from elasticsearch import Elasticsearch

# Elasticsearch 클라이언트 설정 (인증 정보가 필요하다면 추가)
# Elasticsearch 클라이언트 설정
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "changeme")
)  # 본인의 Elasticsearch 주소와 포트를 사용하세요


# 모든 인덱스 목록 가져오기
indices = es.cat.indices(format="json")

# 인덱스 정보 출력
for index in indices:
    print(index['index'])

# 특정 인덱스에서 데이터 조회 (예: image_embeddings 인덱스)
index_name = "image_embeddings"  # 조회할 인덱스 이름
response = es.search(index=index_name, body={"query": {"match_all": {}}}, size=50)


print("------------------")


# 조회된 데이터 출력
for doc in response['hits']['hits']:
    print(f"ID: {doc['_id']}, Source: {doc['_source']}")