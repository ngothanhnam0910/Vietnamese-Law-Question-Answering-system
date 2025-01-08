import os
import json
from tavily import TavilyClient
import openai
from dotenv import load_dotenv, find_dotenv

from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import MessageRole, ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import BaseTool, FunctionTool

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# Thiết lập API key
os.environ["TAVILY_API_KEY"] = "tvly-FED3Gk"

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

search_tool = FunctionTool.from_defaults(fn=search)
llm = OpenAI(model="gpt-4o-mini")
legal_agent = ReActAgent.from_tools([search_tool], llm=llm, verbose=True)

def convert_raw_messages_to_chat_messages(messages):
    """
    Convert a list of messages to a list of ChatMessage instances.

    Args:
        messages (list): List of dictionaries with keys 'role' and 'content'.

    Returns:
        list: List of ChatMessage instances.
    """
    chat_messages = []
    for message in messages:
        role = message.get("role", MessageRole.USER)
        content = message.get("content", "")
        chat_message = ChatMessage.from_str(content=content, role=role)
        chat_messages.append(chat_message)
    return chat_messages


def react_agent_handle(history, question):
    chat_history = convert_raw_messages_to_chat_messages(history)
    response = legal_agent.chat(message=question, chat_history=chat_history)
    return response.response