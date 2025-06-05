from typing import List
from pydantic import BaseModel, Field

class PreferencesRequest(BaseModel):
    uid: str = Field(..., example="1234567890")
    num_conversations: int = Field(2, example=2)

class PreferencesOut(BaseModel):
    preferences: List[str] = Field(..., example=["reading", "traveling"]) 