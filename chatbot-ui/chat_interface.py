import json
import logging
import time
import requests

import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential


st.title("🤖 Chatbot Interface")


# Input bot
bot_id = "botLegal"
user_id = "1"


def send_user_request(text):
    url = f"http://127.0.0.1:8002/chat/complete"

    payload = json.dumps({
        "user_message": text,
        "user_id": str(user_id),
        "bot_id": bot_id,
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
    if response.status_code != 200:
        raise TimeoutError(f"Request to bot fail: {response.text}")
    return json.loads(response.text)


def get_bot_response(request_id):
    url = f"http://127.0.0.1:8002/chat/complete_v2/{request_id}"

    response = requests.request("GET", url, headers={}, data="", timeout=30)
    if response.status_code != 200:
        raise TimeoutError(f"Get bot response fail: {response.text}")
    return response.status_code, json.loads(response.text)


def get_chat_complete(text):
    user_request = send_user_request(text)
    request_id = user_request["task_id"]
    status_code, chat_response = get_bot_response(request_id)
    if status_code == 200:
        print(chat_response)
        return chat_response['task_result']['content']
    else:
        raise TimeoutError("Request fail, try again please")


# Streamed response
def response_generator(user_message):
    res = get_chat_complete(user_message)
    for line in res.split("\n\n"):
        logging.info(f"Line: {line}")
        for sen in line.split("\n"):
            yield sen + '\n\n'
            time.sleep(0.05)
        yield '\n'
    return res


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
