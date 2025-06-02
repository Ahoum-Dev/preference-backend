from pydantic import BaseModel, Field

class NextQuestionIn(BaseModel):
    uid: str = Field(..., example="1234567890")
    num_preferences: int = Field(5, example=5) 