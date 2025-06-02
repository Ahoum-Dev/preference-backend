# Preference Backend

This micro-service ingests seeker-AI conversations, extracts long-term preferences via Graphiti, stores them in Neo4j, and serves dynamic follow-up questions.

## Setup

1. Copy `.env.example` to `.env` and fill in any overrides.
2. Install dependencies:
   ```
   poetry install
   ```
3. Run the API locally:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
4. Or spin up the full stack with Docker Compose:
   ```
   docker-compose up --build
   ```

## Endpoints

- `POST /ingest_conversation` — Ingests a conversation (JSON).
- `POST /next_question` — Returns the next dynamic question.

## Testing

```bash
pytest -q
``` 