import logging
from pymongo import MongoClient
from datetime import datetime

from cache import get_conversation_id
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# MongoDB client setup
client = MongoClient("mongodb://10.0.0.138:27017/")
db = client["final_project"]
chat_conversations = db["history_chat"]

class ChatConversation:
    def __init__(self, conversation_id, bot_id, user_id, message, is_request=True, completed=False, created_at=None, updated_at=None):
        self.conversation_id = conversation_id
        self.bot_id = bot_id
        self.user_id = user_id
        self.message = message
        self.is_request = is_request
        self.completed = completed
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()

    def to_dict(self):
        return {
            "conversation_id": self.conversation_id,
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "message": self.message,
            "is_request": self.is_request,
            "completed": self.completed,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            conversation_id=data["conversation_id"],
            bot_id=data["bot_id"],
            user_id=data["user_id"],
            message=data["message"],
            is_request=data["is_request"],
            completed=data["completed"],
            created_at=data["created_at"],
            updated_at=data["updated_at"]
        )


def load_conversation(conversation_id: str):
    # MongoDB query to load conversations
    conversations = chat_conversations.find({"conversation_id": conversation_id}).sort("created_at")
    return [ChatConversation.from_dict(convo) for convo in conversations]


def read_conversation(conversation_id: str):
    # MongoDB query to get a single conversation
    db_conversation = chat_conversations.find_one({"conversation_id": conversation_id})
    if db_conversation is None:
        raise ValueError("Conversation not found")
    return ChatConversation.from_dict(db_conversation)


def convert_conversation_to_openai_messages(user_conversations):
    conversation_list = [
        {
            "role": "system",
            "content": "You are an amazing virtual assistant"
        }
    ]

    for conversation in user_conversations:
        role = "assistant" if not conversation.is_request else "user"
        content = str(conversation.message)
        conversation_list.append({"role": role, "content": content})

    logging.info(f"Create conversation to {conversation_list}")

    return conversation_list


def update_chat_conversation(bot_id: str, user_id: str, message: str, is_request: bool = True):
    # Step 1: Create a new ChatConversation instance
    conversation_id = get_conversation_id(bot_id, user_id)

    new_conversation = ChatConversation(
        conversation_id=conversation_id,
        bot_id=bot_id,
        user_id=user_id,
        message=message,
        is_request=is_request,
        completed=not is_request,
    )

    # Step 4: Save the ChatConversation instance
    chat_conversations.insert_one(new_conversation.to_dict())

    logger.info(f"Create message for conversation {conversation_id}")

    return conversation_id


def get_conversation_messages(conversation_id):
    user_conversations = load_conversation(conversation_id)
    return convert_conversation_to_openai_messages(user_conversations)
