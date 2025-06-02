from pydantic import BaseModel, Field

class QuestionOut(BaseModel):
    question: str = Field(..., example="What is your favorite movie?") 