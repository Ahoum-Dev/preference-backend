from pydantic import BaseModel, Field

class SummaryRequest(BaseModel):
    uid: str = Field(..., example="1234567890")
    num_conversations: int = Field(2, example=2)

class SummaryOut(BaseModel):
    summary: str 