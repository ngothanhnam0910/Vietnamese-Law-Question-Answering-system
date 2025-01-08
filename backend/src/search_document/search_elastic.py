import json
from elasticsearch import Elasticsearch


# Kết nối tới Elasticsearch
try:
    es = Elasticsearch(
        ["http://localhost:9200"],
    )
    
    # Kiểm tra kết nối
    if es.ping():
        print("Connected to Elasticsearch!")
    else:
        print("Could not connect to Elasticsearch.")
except ConnectionError as e:
    print(f"Error connecting to Elasticsearch: {e}")

# Hàm tìm kiếm dữ liệu trong Elasticsearch
def search_data(index_name, query, top_k=10):
    # Thực hiện tìm kiếm với giới hạn top_k
    response = es.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    "text": query  # Tìm kiếm theo nội dung văn bản
                }
            },
            "sort": [
                {"_score": {"order": "desc"}}  # Sắp xếp theo điểm số giảm dần
            ],
            "size": top_k  # Chỉ định số lượng kết quả muốn lấy ra
        }
    )

    # Lấy kết quả từ response
    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "text": hit["_source"]["text"],
        })

    return results


