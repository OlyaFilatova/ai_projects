# AI Chat Backend Diagrams

This document uses Mermaid diagrams to explain what the project does, how requests move through the system, and how the service is typically run.

## 1. System Overview

```mermaid
flowchart LR
    Client[Client App or curl]
    API[FastAPI API]
    Auth[Auth Routes and Services]
    Chat[Chat Routes and Services]
    DB[(PostgreSQL)]
    LLM["LLM Provider<br/>mock or openai-compatible"]

    Client --> API
    API --> Auth
    API --> Chat
    Auth --> DB
    Chat --> DB
    Chat --> LLM
```

The backend exposes authentication, chat, conversation-history, usage, health, and readiness endpoints. It stores users, refresh tokens, conversations, and messages in PostgreSQL. Chat replies come from a provider adapter, which is either the local mock provider or an OpenAI-compatible upstream accessed through the official OpenAI SDK.

## 2. Application Structure

```mermaid
flowchart LR
    Main[ai_chat.main:app] --> AppFactory[create_app]
    AppFactory --> Container[create_container]
    AppFactory --> ApiFactory[create_api_app]

    Container --> Settings[Settings from AI_CHAT_* env]
    Container --> Migrations[Run Alembic migrations to head]
    Container --> SessionFactory[SQLAlchemy session factory]
    Container --> RateLimiter[Request rate limiter]
    Container --> QuotaLimiter[Per-user quota limiter]

    ApiFactory --> Middleware[HTTP middleware]
    ApiFactory --> Errors[Exception handlers]
    ApiFactory --> AuthRouter["auth router"]
    ApiFactory --> ChatRouter["chat router"]
    ApiFactory --> HealthRouter["health and ready router"]
```

At startup the app loads configuration, runs migrations, prepares the database/session infrastructure, and stores shared dependencies in the application container. Request-scoped services are built from that container.

## 3. Auth and Token Lifecycle

```mermaid
sequenceDiagram
    participant User
    participant AuthAPI as /auth endpoints
    participant UserService
    participant FastAPIUsers as FastAPI Users
    participant DB as PostgreSQL

    User->>AuthAPI: POST /auth/register
    AuthAPI->>UserService: register_user(email, password)
    UserService->>FastAPIUsers: create user
    FastAPIUsers->>DB: insert user
    AuthAPI-->>User: 202 Accepted

    User->>AuthAPI: POST /auth/token
    AuthAPI->>UserService: authenticate(email, password)
    UserService->>FastAPIUsers: verify credentials
    FastAPIUsers->>DB: load user
    UserService->>UserService: create JWT access token
    UserService->>DB: store hashed refresh token
    AuthAPI-->>User: access_token + refresh_token

    User->>AuthAPI: POST /auth/refresh
    AuthAPI->>UserService: refresh_access_token(refresh_token)
    UserService->>DB: look up hashed refresh token
    UserService->>DB: revoke old refresh token
    UserService->>UserService: issue new JWT access token
    UserService->>DB: store new hashed refresh token
    AuthAPI-->>User: new access_token + new refresh_token

    User->>AuthAPI: POST /auth/revoke
    AuthAPI->>UserService: revoke_refresh_token(refresh_token)
    UserService->>DB: mark refresh token revoked
    AuthAPI-->>User: 204 No Content
```

Access tokens are bearer JWTs. Refresh tokens are opaque secrets returned once, stored only as SHA-256 hashes, and rotated on refresh.

## 4. Synchronous Chat Request Flow

```mermaid
sequenceDiagram
    participant User
    participant ChatAPI as POST /chat/messages
    participant Deps as HTTP dependencies
    participant Auth as UserService
    participant ChatService
    participant Quota as Usage quota limiter
    participant Repo as Conversation and Message repos
    participant DB as PostgreSQL
    participant LLM as LLM provider

    User->>ChatAPI: bearer token + message
    ChatAPI->>Deps: resolve dependencies
    Deps->>Auth: read_access_token()
    Auth-->>Deps: current user
    Deps->>ChatService: build chat service
    ChatAPI->>ChatService: reply(user_id, message, conversation_id)
    ChatService->>Quota: enforce per-user chat quota
    ChatService->>Repo: load or create conversation
    Repo->>DB: select/insert conversation
    ChatService->>Repo: store user message
    Repo->>DB: insert message
    ChatService->>Repo: load conversation history
    Repo->>DB: select messages
    ChatService->>LLM: generate_reply(history)
    LLM-->>ChatService: assistant reply
    ChatService->>Repo: store assistant message
    Repo->>DB: insert message
    ChatService-->>ChatAPI: conversation_id + messages
    ChatAPI-->>User: JSON response
```

The sync endpoint persists both the user turn and the final assistant turn before returning the response.

## 5. Streaming Chat Request Flow

```mermaid
sequenceDiagram
    participant User
    participant ChatAPI as POST /chat/messages/stream
    participant Auth as UserService
    participant ChatService
    participant Repo as Message repo
    participant DB as PostgreSQL
    participant LLM as LLM provider
    participant SSE as SSE response

    User->>ChatAPI: bearer token + message
    ChatAPI->>Auth: validate access token
    ChatAPI->>ChatService: stream_reply(user_id, message, conversation_id)
    ChatService->>DB: create conversation if needed
    ChatService->>Repo: store user message
    Repo->>DB: insert message
    ChatService->>LLM: stream_reply(history)
    ChatService-->>SSE: metadata event
    loop for each provider chunk
        LLM-->>ChatService: chunk
        ChatService-->>SSE: chunk event
        SSE-->>User: streamed chunk
    end
    ChatService->>Repo: store concatenated assistant reply
    Repo->>DB: insert assistant message
    ChatService-->>SSE: done event
    SSE-->>User: stream complete
```

The streaming path sends SSE events in this order: `metadata`, repeated `chunk`, then `done`. The assistant message is saved only after the provider stream finishes.

## 6. Request Protection and Error Boundaries

```mermaid
flowchart TD
    Request[Incoming HTTP request]
    BodyLimit[Request size middleware]
    RateLimit[Per-IP rate limit]
    AuthCheck[Bearer token validation for protected routes]
    QuotaCheck[Per-user chat quota]
    Route[Route handler]
    DomainErrors[AppError subclasses]
    HttpErrors[Shared HTTP exception handlers]
    Response[Stable JSON error response]

    Request --> BodyLimit
    BodyLimit --> RateLimit
    RateLimit --> AuthCheck
    AuthCheck --> QuotaCheck
    QuotaCheck --> Route
    Route --> DomainErrors
    DomainErrors --> HttpErrors
    HttpErrors --> Response
```

The app rejects oversized bodies early, rate-limits high-risk routes by client IP, enforces per-user chat quotas for authenticated chat actions, and maps domain errors into stable JSON responses with `detail` and `code`.

## 7. How Developers Use the Project Locally

```mermaid
flowchart TD
    Clone[Clone repository]
    Venv[Create virtualenv]
    Install["pip install -e .[dev]"]
    Env[Set AI_CHAT_* environment variables]
    Tests[Run pytest]
    Run[Run uvicorn ai_chat.main:app --reload]
    Register[Register and sign in]
    Call[Call /chat and /auth endpoints]

    Clone --> Venv --> Install --> Env
    Env --> Tests
    Env --> Run
    Run --> Register --> Call
```

Typical local usage is: install dependencies, configure `AI_CHAT_DATABASE__URL`, `AI_CHAT_JWT__SECRET`, and `AI_CHAT_LLM__PROVIDER`, run tests, start Uvicorn, then exercise the API with `curl`, a frontend, or an API client.

## 8. Docker and Deployment Topology

### Local Compose

```mermaid
flowchart LR
    User[Developer]
    API["api service<br/>FastAPI container"]
    DB["db service<br/>PostgreSQL"]

    User -->|http://localhost:8000| API
    API --> DB
```

`docker-compose.yml` runs two services: the API and PostgreSQL.

### Production Compose

```mermaid
flowchart LR
    Internet[Client / Internet]
    Proxy[Nginx proxy]
    API["api service<br/>FastAPI"]
    DB["db service<br/>PostgreSQL"]
    Upstream[Optional upstream LLM API]

    Internet --> Proxy
    Proxy --> API
    API --> DB
    API --> Upstream
```

`docker-compose.prod.yml` puts Nginx in front of the API. The proxy handles external traffic, forwards headers, preserves SSE compatibility, and adds an extra deployment-layer request/connection guard before requests reach FastAPI.
