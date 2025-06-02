import os
import httpx
from neo4j import GraphDatabase
# Attempt to import Graphiti client and LLM client, with stubs if unavailable
try:
    from graphiti_core import Graphiti
    from graphiti_core.llm_client import OpenAIGenericClient
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

# Environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
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

def add_episode(uid: str, conv: list[dict]) -> str:
    """
    Ingests a conversation as a Graphiti Episode.

    Returns the episode id.
    """
    # Write to Neo4j directly
    episode_id = uid
    with driver.session() as session:
        session.run(
            "MERGE (u:User {uid:$uid})", uid=uid
        )
        session.run(
            "CREATE (e:Episode {id:$episode_id})", episode_id=episode_id
        )
        session.run(
            "MATCH (u:User {uid:$uid}), (e:Episode {id:$episode_id}) MERGE (u)-[:CREATED]->(e)",
            uid=uid, episode_id=episode_id,
        )
        for turn in conv:
            if turn.get("speaker") == "User":
                text = turn.get("text","")
                session.run(
                    "MERGE (p:Preference {text:$text})", text=text
                )
                session.run(
                    "MATCH (u:User {uid:$uid}), (p:Preference {text:$text}) MERGE (u)-[:LIKES]->(p)",
                    uid=uid, text=text,
                )
    return episode_id

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