import os
from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger

load_dotenv()

from app.routes.ingest import router as ingest_router
from app.routes.questions import router as questions_router
from app.routes.summary import router as summary_router

app = FastAPI(title="Preference Backend")

app.include_router(ingest_router)
app.include_router(questions_router) 
app.include_router(summary_router) 