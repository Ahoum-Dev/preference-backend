import os
from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger

load_dotenv()

from app.routes.ingest import router as ingest_router
from app.routes.questions import router as questions_router
from app.routes.summary import router as summary_router
from app.routes.content import router as content_router
from app.routes.preferences import router as preferences_router
from app.routes.conversation_summary import router as conversation_summary_router
from app.routes.get_conversation import router as get_conversation_router

app = FastAPI(title="Preference Backend")

app.include_router(ingest_router)
app.include_router(questions_router) 
app.include_router(summary_router) 
app.include_router(preferences_router) 
app.include_router(content_router) 
app.include_router(conversation_summary_router) 
app.include_router(get_conversation_router)