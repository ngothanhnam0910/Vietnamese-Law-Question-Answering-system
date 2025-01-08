import os
import json
from tavily import TavilyClient
import openai
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def search(query):
    # get 3 first links
    output_search = tavily_client.search(query).get('results')[:3]
    # Xử lý kết quả tìm kiếm thành chuỗi tài liệu
    search_document = "Dưới đây là các tài liệu truy xuất được từ internet: \n"
    for i, doc in enumerate(output_search):
        search_document += f"{i+1}. {doc.get('content', '')} \n"
    search_document += "Kết thúc phần tài liệu truy xuất được."
    return search_document

# define a function
functions_info = [
    {
        "name": "search",
        "description": "Get information in internet base on user query ",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "This is user query",
                },
            },
            "required": ["query"],
        },
    }
]

