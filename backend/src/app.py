import logging
import time
from typing import Dict, Optional
from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from utils import setup_logging
from tasks import llm_handle_message

from search_document.combine_search import CombinedSearch
from search_document.rerank import BGEReranker

setup_logging()
logger = logging.getLogger(__name__)

# init retriever and reranker
combined_search_instance = CombinedSearch()
reranker_instance = BGEReranker(model_name="/home/ivirse/ivirse_all_data/namnt/soict/checkpoint/rerank/bge_v2_part2/checkpoint-225000", use_fp16=True)


app = FastAPI()

# define class name
class CompleteRequest(BaseModel):
    bot_id: Optional[str] = 'bot_Legal_VN'
    user_id: str
    user_message: str
    sync_request: Optional[bool] = False

class RetrievalRequest(BaseModel):
    query: str
    top_k_search: int = 30
    top_k_rerank: int = 5


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/retrieval")
async def retrieval(request: RetrievalRequest):
    try:
        # Lấy dữ liệu từ body
        query = request.query
        top_k_search = request.top_k_search
        top_k_rerank = request.top_k_rerank
        # Thực hiện tìm kiếm bằng CombinedSearch
        search_results = combined_search_instance.search(query_text=query, top_k=top_k_search)

        # Thực hiện rerank kết quả tìm kiếm
        reranked_results = reranker_instance.rerank(query=query, documents=search_results, topk=top_k_rerank)

        return {
            "results": reranked_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@app.post("/chat/complete")
async def complete(data: CompleteRequest):
    bot_id = data.bot_id
    user_id = data.user_id
    user_message = data.user_message
    logger.info(f"Complete chat from user {user_id} to {bot_id}: {user_message}")

    if not user_message or not user_id:
        raise HTTPException(status_code=400, detail="User id and user message are required")

    if data.sync_request:
        response = llm_handle_message(bot_id, user_id, user_message)
        return {"response": str(response)}
    else:
        task = llm_handle_message.delay(bot_id, user_id, user_message)
        return {"task_id": task.id}


@app.get("/chat/complete_v2/{task_id}")
async def get_response(task_id: str):
    start_time = time.time()
    timeout = 60  # Timeout sau 60 giây
    polling_interval = 0.1  # Thời gian chờ giữa mỗi lần kiểm tra (100ms)
    
    while True:
        # Lấy trạng thái task từ Celery
        task_result = AsyncResult(task_id)
        task_status = task_result.status
        
        # Ghi log trạng thái task
        logger.info(f"Task ID: {task_id}, Status: {task_status}")
        
        # Nếu task đã hoàn tất, trả về kết quả
        if task_status not in ('PENDING', 'STARTED'):
            return {
                "task_id": task_id,
                "task_status": task_status,
                "task_result": task_result.result
            }
        
        # Kiểm tra timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            logger.warning(f"Task {task_id} timed out after {timeout} seconds.")
            return {
                "task_id": task_id,
                "task_status": task_status,
                "error_message": "Service timeout, please retry."
            }
        
        # Chờ trước khi kiểm tra lại
        await asyncio.sleep(polling_interval)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8002, workers=1, log_level="info")

