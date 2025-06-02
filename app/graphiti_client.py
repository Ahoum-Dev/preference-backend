import os
import httpx
import json
import re
from neo4j import GraphDatabase
from loguru import logger
# Attempt to import Graphiti client and LLM client, with stubs if unavailable
try:
    from graphiti_core import Graphiti
    from graphiti_core.llm_client import OpenAIGenericClient
    _USE_GRAPHITI = True
except ImportError:
    class OpenAIGenericClient:
        def __init__(self, *args, **kwargs):
            pass

    class Graphiti:
        def __init__(self, *args, **kwargs):
            pass

        def add_episode(self, uid, conversation):
            # stub that returns dummy episode with id equal to uid
            class Episode:
                id = uid
            return Episode()
    _USE_GRAPHITI = False

# Environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("NEBIUS_MODEL_NAME")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
GRAPHITI_LLM_TIMEOUT = int(os.getenv("GRAPHITI_LLM_TIMEOUT", "25"))

# Initialize LLM client
llm_client = OpenAIGenericClient(
    base_url=OPENAI_API_BASE,
    api_key=OPENAI_API_KEY,
    model_name=MODEL_NAME,
    timeout=GRAPHITI_LLM_TIMEOUT,
)

# Initialize Neo4j driver
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
)

# Initialize Graphiti client if available
if _USE_GRAPHITI:
    graphiti = Graphiti(driver=driver, llm_client=llm_client, embedding_model_name=EMBEDDING_MODEL_NAME)
else:
    graphiti = None

def add_episode(uid: str, conv: list[dict]) -> str:
    """
    Ingests a conversation as a Graphiti Episode and extracts multiple relationships.

    Returns the episode id.
    """
    logger.info(f"add_episode called with uid={uid}, num_turns={len(conv)}, USE_GRAPHITI={_USE_GRAPHITI}")
    if not _USE_GRAPHITI:
        logger.info(f"Using fallback manual ingestion for uid={uid}")
        with driver.session() as session:
            # Log raw user texts
            logger.debug(f"User turns: {[t.get('text','') for t in conv if t.get('speaker')=='User']}")
            # Ensure user node exists
            session.run("MERGE (u:User {uid:$uid})", uid=uid)
            # Prepare user-only text for LLM
            conv_text = "\n".join([t.get("text", "") for t in conv if t.get("speaker") == "User"])
            logger.debug(f"conv_text for LLM: {conv_text}")
            # Ask LLM to extract relationships
            logger.info(f"Sending LLM request to {OPENAI_API_BASE}/chat/completions")
            chat_payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "Extract relationships from this user conversation. Output JSON list of objects with fields 'relation', 'object', and 'object_type'."},
                    {"role": "user", "content": conv_text},
                ],
                "max_tokens": 500,
                "temperature": 0,
            }
            try:
                resp = httpx.post(
                    f"{OPENAI_API_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json=chat_payload,
                )
            except Exception as e:
                logger.error(f"LLM request failed for uid={uid}: {e}")
                raise
            resp.raise_for_status()
            logger.debug(f"LLM response status: {resp.status_code}")
            content = resp.json()["choices"][0]["message"]["content"]
            logger.debug(f"LLM response content: {content}")
            # Extract JSON array from LLM output
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]
                logger.debug(f"Extracted JSON string for parsing: {json_str}")
                try:
                    rels = json.loads(json_str)
                except Exception as e:
                    logger.error(f"JSON parse error: {e}; json_str: {json_str}")
                    rels = []
            else:
                logger.error(f"No JSON array found in LLM output: {content}")
                rels = []
            logger.debug(f"Extracted relationships: {rels}")
            # Create nodes and relationships based on LLM output
            rel_count = 0
            for rel in rels:
                # Sanitize relationship type: replace non-alphanumeric with underscore
                raw_rel = rel.get("relation", "REL")
                rel_type = re.sub(r"\W+", "_", raw_rel).upper()
                # Sanitize object label (capitalize, no spaces)
                raw_obj_type = rel.get("object_type", "Preference")
                obj_type = raw_obj_type.strip().replace(' ', '_').capitalize()
                obj = rel.get("object", "")
                logger.info(f"Creating relationship {uid}-[:{rel_type}]->{obj_type}({obj})")
                # Merge object node
                session.run(f"MERGE (o:{obj_type} {{name:$obj}})", obj=obj)
                # Link user to object with sanitized rel_type
                session.run(
                    f"MATCH (u:User {{uid:$uid}}), (o:{obj_type} {{name:$obj}}) MERGE (u)-[:{rel_type}]->(o)",
                    uid=uid, obj=obj,
                )
                rel_count += 1
            logger.info(f"Total relationships created for uid={uid}: {rel_count}")
        return uid
    # Use Graphiti to ingest conversation and extract relationships into Neo4j
    logger.info(f"Using Graphiti ingestion for uid={uid}")
    try:
        episode = graphiti.add_episode(uid=uid, conversation=conv)
    except Exception as e:
        logger.error(f"Graphiti.add_episode failed for uid={uid}: {e}")
        raise
    logger.info(f"Graphiti.add_episode returned episode_id={episode.id}")
    return episode.id

# Add LLM-based question generation and Graphiti preference search
SYSTEM_PROMPT = "You are a helpful assistant that generates the next follow-up question to learn about user preferences. Only output the question."

async def generate_next_question(preferences: list[str]) -> str:
    """
    Generate the next dynamic question given existing preferences using LLM,
    trying chat/completions first, then falling back to completions endpoint.
    """
    pref_text = ", ".join(preferences) if preferences else ""
    if pref_text:
        user_prompt = f"User preferences so far: {pref_text}. Ask the next question to learn another preference."
    else:
        user_prompt = "Ask a question to learn about the user's preferences."
    async with httpx.AsyncClient(timeout=GRAPHITI_LLM_TIMEOUT) as client:
        # First, try chat/completions
        chat_payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            resp = await client.post(
                f"{OPENAI_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json=chat_payload,
            )
            resp.raise_for_status()
            chat_data = resp.json()
            return chat_data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as chat_err:
            if chat_err.response.status_code != 404:
                raise
        # Fallback to legacy completions
        comp_payload = {
            "model": MODEL_NAME,
            "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
            "max_tokens": 64,
        }
        resp2 = await client.post(
            f"{OPENAI_API_BASE}/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json=comp_payload,
        )
        resp2.raise_for_status()
        comp_data = resp2.json()
        return comp_data.get("choices", [{}])[0].get("text", "").strip()

def get_preferences(uid: str, top_k: int = 5) -> list[str]:
    """
    Retrieve top_k preferences for a user using Graphiti hybrid search recipe.
    """
    # Fetch preferences from Neo4j
    with driver.session() as session:
        result = session.run(
            "MATCH (u:User {uid:$uid})-[:LIKES]->(p:Preference) RETURN p.text AS text LIMIT $k",
            uid=uid, k=top_k,
        )
        return [record["text"] for record in result] 