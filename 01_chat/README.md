# AI Chat Backend

This repository contains a working AI chat backend built around FastAPI, PostgreSQL, SQLAlchemy, and Docker. The current codebase supports local user registration, JWT authentication, persisted conversations and messages, synchronous chat replies, SSE streaming, provider selection, Alembic migrations, structured request logging, and a small set of conversation/usage endpoints.

## What works today

- FastAPI application factory and entry point
- explicit configuration loading and dependency wiring
- health and readiness endpoints
- SQLAlchemy persistence layer for users, conversations, and messages
- FastAPI-Users-backed local registration, password hashing, JWT access tokens, and bearer-token auth dependency
- refresh-token rotation and explicit refresh-token revocation
- authenticated synchronous and streaming chat endpoints
- conversation listing, history retrieval, and per-user usage reporting
- provider abstraction with mock and official OpenAI-SDK-backed OpenAI-compatible adapter
- conservative provider retry/timeout handling with centralized provider error mapping
- `slowapi`/`limits`-backed rate limiting for high-risk auth and chat write endpoints
- `limits`-backed per-user usage quotas for synchronous and streaming chat actions
- `sse-starlette`-backed SSE transport for streamed chat responses
- Alembic migration scaffolding with an initial schema revision
- structured JSON-style request logging and request-size guardrails
- Docker image and Compose setup for local API + PostgreSQL startup
- production-oriented Docker Compose deployment file
- Nginx reverse-proxy layer for production Compose with SSE-safe proxying and request limits
- pytest-based coverage for configuration loading, startup behavior, repository behavior, auth flows, chat flows, migrations, and provider adapters

## What does not work yet

- richer user-management features beyond local registration and token lifecycle
- conversation deletion, renaming, or archival controls
- live upstream verification of the OpenAI-SDK-backed provider path in tests
- distributed tracing or advanced multi-instance edge protections
- shared-store rate limiting across multiple API instances

## Local development

1. Create and activate a virtual environment.
2. Install the package with development dependencies:

```bash
python3 -m pip install -e ".[dev]"
```

3. Run tests:

```bash
pytest
```

The test suite does not require the API server to be running. It uses in-process FastAPI test clients and isolated test databases.

4. Run the API when you want to exercise the service manually:

```bash
uvicorn ai_chat.main:app --reload
```

5. Call the health endpoint once the app is running:

```bash
curl http://127.0.0.1:8000/health
```

6. If you want to apply migrations manually before startup, run:

```bash
alembic upgrade head
```

## Environment variables

Current configuration uses the `AI_CHAT_` prefix and nested keys with `__`.

Recommended for local development:
- `AI_CHAT_DATABASE__URL=postgresql+psycopg://postgres:<db-password>@localhost:5432/ai_chat`
- `AI_CHAT_JWT__SECRET=<long-random-secret>`
- `AI_CHAT_LLM__PROVIDER=mock`

Required in real deployments:
- `AI_CHAT_ENV`: `development`, `test`, or `production`
- `AI_CHAT_DATABASE__URL`: PostgreSQL connection string
- Example: `postgresql+psycopg://postgres:postgres@localhost:5432/ai_chat`
- `AI_CHAT_JWT__SECRET`: JWT signing secret
- `AI_CHAT_JWT__ALGORITHM`: current default is `HS256`
- `AI_CHAT_JWT__ACCESS_TOKEN_EXPIRE_MINUTES`: access-token lifetime
- `AI_CHAT_JWT__REFRESH_TOKEN_EXPIRE_DAYS`: refresh-token lifetime
- `AI_CHAT_LLM__PROVIDER`: current provider selector, default `mock`
- `AI_CHAT_LLM__TIMEOUT_SECONDS`: provider timeout budget
- `AI_CHAT_LLM__MAX_RETRIES`: conservative retry count for provider calls
- `AI_CHAT_LLM__MODEL`: model name for provider integrations
- `AI_CHAT_MAX_REQUEST_BODY_BYTES`: request-size guardrail for incoming HTTP bodies
- `AI_CHAT_RATE_LIMIT__STORAGE_URI`: rate-limit backend URI, default `memory://`
- `AI_CHAT_USAGE_QUOTA__STORAGE_URI`: quota backend URI, default `memory://`

Optional placeholders already supported:
- `AI_CHAT_DEBUG`
- `AI_CHAT_LLM__API_BASE_URL`
- `AI_CHAT_LLM__API_KEY`
- `AI_CHAT_LOG_LEVEL`

## Authentication

Current auth endpoints:
- `POST /auth/register`
- `POST /auth/token`
- `POST /auth/refresh`
- `POST /auth/revoke`
- `GET /auth/me`

Users are stored locally, and FastAPI Users now provides the primary local-user manager, password hashing, and JWT access-token strategy through an async auth path. Protected routes still accept only access tokens. Refresh tokens remain application-specific opaque random secrets, stored only as SHA-256 hashes, and can be rotated or revoked through dedicated endpoints. Public registration responses are intentionally generic so duplicate-account attempts do not reveal whether an email already exists. Roles, account recovery, and broader user-management features are still not implemented.

## Chat

Current chat endpoint:
- `POST /chat/messages`
- `POST /chat/messages/stream`
- `GET /chat/conversations`
- `GET /chat/conversations/{conversation_id}`
- `GET /chat/usage`

Requests must include a bearer token. The synchronous endpoint accepts a user message and an optional `conversation_id`, stores the user and assistant turns, and returns a complete assistant message in the response body. The streaming endpoint is implemented with `sse-starlette` and emits Server-Sent Events in this order:
- `metadata`: conversation metadata for the request
- `chunk`: assistant content chunks
- `done`: completion marker after the full reply is streamed

Supported provider paths:
- `mock`: deterministic local provider for tests and development
- `openai-compatible`: official OpenAI Python SDK pointed at an OpenAI-style chat-completions API

Example request:

```bash
curl -X POST http://127.0.0.1:8000/chat/messages \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello there"}'
```

Example SSE stream request:

```bash
curl -N -X POST http://127.0.0.1:8000/chat/messages/stream \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello there"}'
```

## Architecture

Current boundaries:
- `transport/http`: FastAPI routers and request-bound dependencies
- `services`: authentication and chat orchestration logic
- `repositories`: persistence access for users, conversations, and messages
- `adapters/llm`: provider adapters behind the current LLM boundary

The internal wiring is intentionally lightweight: the application container keeps shared startup-time state, while request-bound dependencies build only the small number of request-scoped services that still need per-request sessions.

## Conversation controls

Current conversation support includes:
- listing the current user’s conversations
- retrieving the full stored history of one conversation
- retrieving per-user usage counts for conversations and messages

## Error handling

Current domain errors are mapped through shared HTTP exception handlers and return:
- `detail`: human-readable message
- `code`: stable machine-friendly error code

Examples already covered by tests include invalid credentials, missing auth headers, missing conversations, and provider unavailability.
Timeouts, SDK request failures, and transient provider failures are mapped through the same reliability layer before reaching the shared API error-handling path.
Oversized request bodies are rejected early with a stable `request_too_large` error.
Registration, login, synchronous chat, and streaming chat creation also return `rate_limit_exceeded` when the per-IP budget is exhausted.
Authenticated users also receive `quota_exceeded` when they exceed the configured per-user chat budget.
Refresh endpoints return `auth_refresh_token_invalid` for expired, revoked, unknown, or already-rotated refresh tokens.
Registration returns the same public `202 Accepted` response for first-time and duplicate submissions to reduce account-enumeration signals.
Request-rate limits and chat usage quotas are both enforced through a small shared fixed-window helper built on `limits`, with `slowapi` still used only for client-address extraction on HTTP rate limits.

For local development and tests, the JWT secret can be omitted. Production mode already requires `AI_CHAT_JWT__SECRET`.

No real secrets are stored in the repository or baked into the container image. Docker-based workflows should provide values through environment variables or env files that are kept outside version control.

## Docker

Start the local stack:

1. Copy `.env.example` to a local env file and replace the placeholder secrets.
2. Start the stack:

```bash
cp .env.example .env
docker compose --env-file .env up --build
```

The Compose file includes:
- `api`: FastAPI application
- `db`: PostgreSQL database for the current persistence layer

## Deployment

The repository now includes a production-oriented Compose definition in `docker-compose.prod.yml`.

Basic workflow:

1. Copy `.env.production.example` to your deployment environment and replace the secrets.
2. Build and start the stack:

```bash
docker compose -f docker-compose.prod.yml --env-file .env.production.example up --build -d
```

Notes:
- the image includes `alembic.ini` and `migrations/`, so startup-time migrations work inside the container
- the production Compose file runs Nginx in front of the API for request-size limits, proxy headers, and SSE-compatible connection handling
- the production Compose file uses restart policies and health checks for the proxy, API, and PostgreSQL
- the database is not published on a host port by default in the production Compose file
- secrets are injected through environment variables instead of being copied into the image

Proxy behavior in production Compose:
- external traffic reaches the `proxy` service on `${AI_CHAT_PORT:-8000}`
- the `api` service is only exposed on the internal Compose network
- Nginx forwards standard `X-Forwarded-*` headers to FastAPI
- request bodies are capped at `16k` at the proxy, matching the current application default
- per-IP connection and request limits are applied at the proxy as a first deployment-layer guardrail
- buffering is disabled for upstream responses so `/chat/messages/stream` remains compatible with SSE

## Operations

Current operational endpoints:
- `GET /health`: basic liveness information
- `GET /ready`: readiness check including database connectivity

Current logging:
- request logs are emitted in a structured JSON format

## Migrations

Alembic is configured under `migrations/`.

Useful commands:

```bash
alembic upgrade head
alembic downgrade -1
```

Current note:
- the app applies Alembic migrations during startup, including normal container boot, so the migration path is the active schema-management path

## Persistence

The application now includes SQLAlchemy models and repositories for:
- users
- conversations
- messages
- refresh tokens

PostgreSQL remains the intended runtime database through `AI_CHAT_DATABASE__URL`. The test suite uses isolated in-memory SQLite engines for fast repository verification.

## Limitations

- streaming exists only as SSE
- the mock provider is what the test suite exercises end to end
- the real provider path now uses the official OpenAI Python SDK, but it is still not exercised against a live upstream in the test suite
- interrupted streams keep the user message but do not persist a partial assistant message
- SSE formatting now relies on `sse-starlette`, while event names and payload shapes remain application-defined
- retries are conservative and currently intended only for failures before a complete response is emitted
- the OpenAI-compatible adapter is intentionally thin, while provider-facing error translation now lives in the reliability wrapper
- rate limiting and usage quotas default to in-memory storage unless you configure a shared `limits` backend URI
- usage quotas are intended as a lightweight abuse/spend control, not a billing system
- refresh-token state is stored locally in the primary database and does not yet include device/session management features
- FastAPI Users now handles most auth internals through an async path, but the project still keeps custom public auth routes and a custom opaque refresh-token lifecycle
- production deployments should use long JWT secrets and https provider base URLs
- current proxy protections are static Nginx limits, not a distributed or identity-aware edge policy
