from fastapi import APIRouter, HTTPException
from app.models.summary import SummaryRequest, SummaryOut
import app.graphiti_client as graphiti_client
import httpx
import json
from loguru import logger

# Shortcut to LLM settings
OPENAI_API_BASE = graphiti_client.OPENAI_API_BASE
OPENAI_API_KEY = graphiti_client.OPENAI_API_KEY
MODEL_NAME = graphiti_client.MODEL_NAME
TIMEOUT = graphiti_client.GRAPHITI_LLM_TIMEOUT

router = APIRouter()

@router.post("/conversation_summary", response_model=SummaryOut)
async def conversation_summary(payload: SummaryRequest):
    """
    Summarize the last `num_conversations` for a given user.
    """
    try:
        # Use Graphiti core summarization if available
        if getattr(graphiti_client, '_USE_GRAPHITI', False) and graphiti_client.graphiti:
            # Use Graphiti core summarization if available
            summary = await graphiti_client.graphiti.summarize_episodes(
                uid=payload.uid,
                num_conversations=payload.num_conversations
            )
        else:
            # Fallback: summarize using LLM and stored conversations in Neo4j
            with graphiti_client.driver.session() as session:
                result = session.run(
                    "MATCH (u:User {uid:$uid})-[:CREATED]->(e:Episode) "
                    "RETURN e.conversation AS conv_json ORDER BY e.created_at DESC LIMIT $n",
                    uid=payload.uid, n=payload.num_conversations
                )
                convs = [json.loads(record["conv_json"]) for record in result]
            if not convs:
                summary = f"No conversations found for user {payload.uid}."
            else:
                # Flatten to text
                text_blocks = []
                for conv in convs:
                    lines = [f"{turn.get('speaker')}: {turn.get('text','')}" for turn in conv]
                    text_blocks.append("\n".join(lines))
                convo_text = "\n\n".join(text_blocks)
                # Build LLM prompt
                prompt = f"Summarize the following conversations:\n\n{convo_text}\n\nProvide a concise summary."
                async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                    resp = await client.post(
                        f"{OPENAI_API_BASE}/chat/completions",
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                        json={
                            "model": MODEL_NAME,
                            "messages": [
                                {"role": "system", "content": "You are a helpful summarization assistant."},
                                {"role": "user", "content": prompt},
                            ],
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    summary = data.get("choices", [])[0].get("message", {}).get("content", "").strip()
        return SummaryOut(summary=summary)
    except Exception as e:
        logger.error(f"Error in conversation_summary for uid={payload.uid}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 