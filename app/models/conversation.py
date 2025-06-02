from typing import List
from datetime import datetime
from pydantic import BaseModel, Field

class Turn(BaseModel):
    speaker: str = Field(..., example="AI")
    text: str = Field(..., example="Hello!")

class ConversationIn(BaseModel):
    uid: str = Field(..., example="1234567890")
    conversation: List[Turn]
    conversation_id: str
    created_at: datetime
    updated_at: datetime 