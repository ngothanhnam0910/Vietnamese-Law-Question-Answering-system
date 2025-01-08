from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class ChatConversationCreate(BaseModel):
    bot_id: str
    user_id: str
    message: str
    is_request: Optional[bool] = True
    completed: Optional[bool] = False

class ChatConversation(BaseModel):
    conversation_id: str
    bot_id: str
    user_id: str
    message: str
    is_request: bool
    completed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
