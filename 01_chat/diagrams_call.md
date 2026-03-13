# Function And Class Call Diagrams

This document focuses on code-level interactions: which functions and classes call each other, and when those calls happen during startup and request handling.

## 1. Application Startup Call Graph

```mermaid
flowchart LR
    A["ai_chat.main:app"] --> B["create_app()"]
    B --> C["create_container()"]
    B --> D["create_api_app()"]

    C --> E["build_container()"]
    E --> F["run_migrations_to_head()"]
    E --> G["create_engine_from_settings()"]
    E --> H["create_session_factory()"]
    E --> I["FixedWindowLimitEngine(...)"]

    D --> J["configure_logging()"]
    D --> K["FastAPI(...)"]
    D --> L["register_exception_handlers()"]
    D --> M["register_http_middleware()"]
    D --> N["include auth router"]
    D --> O["include chat router"]
    D --> P["include health router"]
```

When the process starts, `create_app()` builds the shared container first, then builds the FastAPI app around that container.

## 2. Request Dependency Resolution

```mermaid
flowchart TD
    A["Incoming request"] --> B["get_container(request)"]
    B --> C["ServiceContainer"]

    C --> D["get_db_session(container)"]
    D --> E["session_factory()"]
    E --> F["Session"]

    F --> G["get_user_service(session, container)"]
    F --> H["get_chat_service(session, container)"]

    G --> I["UserService(...)"]
    H --> J["build_provider(settings)"]
    J --> K["ReliableLlmProvider(...)"]
    H --> L["ChatService(...)"]

    G --> M["get_current_user(credentials, user_service)"]
    M --> N["user_service.read_access_token()"]
```

This is the shared path FastAPI uses before the route handler itself runs.

## 3. Auth Route Call Flow

### Register Route To UserService Boundary

```mermaid
sequenceDiagram
    participant Client
    participant Router as auth router
    participant Dep as get_user_service
    participant Service as UserService

    Client->>Router: POST /auth/register
    Router->>Dep: build UserService
    Dep-->>Router: UserService
    Router->>Service: register_user(email, password)
```

### Inside UserService.register_user()

```mermaid
sequenceDiagram
    participant Service as UserService
    participant Manager as AiChatUserManager
    participant UserDB as SyncUserDatabase
    participant Users as UserRepository

    Service->>Manager: create(UserCreate(...))
    Manager->>UserDB: get_by_email(...)
    UserDB->>Users: get_by_email(...)
    Manager->>UserDB: create(...)
    UserDB->>Users: persist UserModel
```

### Token Route To UserService Boundary

```mermaid
sequenceDiagram
    participant Client
    participant Router as auth router
    participant Service as UserService

    Client->>Router: POST /auth/token
    Router->>Service: authenticate(email, password)
```

### Inside UserService.authenticate()

```mermaid
sequenceDiagram
    participant Service as UserService
    participant Manager as AiChatUserManager
    participant UserDB as SyncUserDatabase
    participant Users as UserRepository
    participant Refresh as RefreshTokenRepository
    participant JWT as JWTStrategy

    Service->>Manager: authenticate(OAuth2PasswordRequestForm)
    Manager->>UserDB: get_by_email(...)
    UserDB->>Users: get_by_email(...)
    Service->>Service: _issue_token_pair(user_id)
    Service->>Service: _get_access_token_strategy()
    Service->>JWT: write_token(user)
    Service->>Refresh: create(user_id, token_hash, expires_at)
```

`/auth/refresh` and `/auth/revoke` reuse the same `UserService` but go through `RefreshTokenRepository` instead of the credential-authentication path.

## 4. Refresh And Revoke Call Flow

### Refresh Token Flow

```mermaid
sequenceDiagram
    participant Client
    participant Router as auth router
    participant Service as UserService
    participant Refresh as RefreshTokenRepository
    participant Manager as AiChatUserManager
    participant JWT as JWTStrategy

    Client->>Router: POST /auth/refresh
    Router->>Service: refresh_access_token(refresh_token)
    Service->>Service: _get_refresh_token(...)
    Service->>Service: _hash_refresh_token(...)
    Service->>Refresh: get_by_hash(token_hash)
    Service->>Service: _normalize_timestamp(...)
    Service->>Manager: get(user_id)
    Service->>Refresh: revoke(old token)
    Service->>Service: _issue_token_pair(user_id)
    Service->>JWT: write_token(user)
    Service->>Refresh: create(new token)
    Router-->>Client: new token pair
```

### Revoke Token Flow

```mermaid
sequenceDiagram
    participant Client
    participant Router as auth router
    participant Service as UserService
    participant Refresh as RefreshTokenRepository

    Client->>Router: POST /auth/revoke
    Router->>Service: revoke_refresh_token(refresh_token)
    Service->>Service: _get_refresh_token(...)
    Service->>Refresh: revoke(token)
    Router-->>Client: 204
```

The refresh and revoke endpoints share the same internal refresh-token lookup path.

## 5. Synchronous Chat Call Flow

### Route To ChatService Boundary

```mermaid
sequenceDiagram
    participant Client
    participant Router as chat router
    participant Deps as auth and chat dependencies
    participant UserSvc as UserService
    participant ChatSvc as ChatService

    Client->>Router: POST /chat/messages
    Router->>Deps: limit_chat_requests(...)
    Router->>Deps: get_current_user(...)
    Deps->>UserSvc: read_access_token(token)
    Router->>Deps: get_chat_service(...)
    Deps->>ChatSvc: ChatService(...)
    Router->>ChatSvc: reply(user_id, message, conversation_id)
```

### Inside ChatService.reply()

```mermaid
sequenceDiagram
    participant ChatSvc as ChatService
    participant Quota as enforce_usage_quota()
    participant ConvRepo as ConversationRepository
    participant MsgRepo as MessageRepository
    participant Provider as ReliableLlmProvider
    participant Base as Base provider

    ChatSvc->>ChatSvc: _normalize_message(...)
    ChatSvc->>Quota: enforce_usage_quota(...)
    ChatSvc->>ChatSvc: _load_or_create_conversation(...)
    ChatSvc->>ConvRepo: create(...) or get_for_user(...)
    ChatSvc->>ChatSvc: _store_user_message_and_build_history(...)
    ChatSvc->>MsgRepo: create(user message)
    ChatSvc->>ChatSvc: _build_history(...)
    ChatSvc->>MsgRepo: list_for_conversation(...)
    ChatSvc->>Provider: generate_reply(messages)
    Provider->>Base: generate_reply(messages)
    Base-->>Provider: assistant text
    Provider-->>ChatSvc: assistant text
    ChatSvc->>MsgRepo: create(assistant message)
```

The first diagram ends when the route calls `ChatService.reply(...)`. The second shows the internal `ChatService` work: validation, quota enforcement, conversation lookup or creation, message persistence, provider invocation, and assistant-message persistence.

## 6. Streaming Chat Call Flow

```mermaid
sequenceDiagram
    participant Client
    participant Router as chat router
    participant ChatSvc as ChatService
    participant MsgRepo as MessageRepository
    participant Provider as ReliableLlmProvider
    participant Base as Base provider
    participant SSE as build_chat_sse_events() and create_sse_response()

    Client->>Router: POST /chat/messages/stream
    Router->>ChatSvc: stream_reply(user_id, message, conversation_id)
    ChatSvc->>ChatSvc: _normalize_message(...)
    ChatSvc->>ChatSvc: _load_or_create_conversation(...)
    ChatSvc->>ChatSvc: _store_user_message_and_build_history(...)
    ChatSvc->>MsgRepo: create(user message)
    ChatSvc->>MsgRepo: list_for_conversation(...)
    ChatSvc-->>Router: StreamingChatResult(chunk_iterator)

    Router->>SSE: build_chat_sse_events(...)
    Router->>SSE: create_sse_response(...)
    SSE-->>Client: metadata event

    loop while streaming
        ChatSvc->>Provider: stream_reply(messages)
        Provider->>Base: stream_reply(messages)
        Base-->>Provider: chunk
        Provider-->>ChatSvc: chunk
        ChatSvc-->>SSE: yielded chunk
        SSE-->>Client: chunk event
    end

    ChatSvc->>MsgRepo: create(full assistant message)
    SSE-->>Client: done event
```

The assistant message is persisted only after the chunk iterator finishes collecting the full stream.

## 7. Provider Construction Call Graph

```mermaid
flowchart LR
    A["get_chat_service(...)"] --> B["build_provider(settings)"]
    B --> C["_build_base_provider(settings)"]

    C --> D["MockLlmProvider()"]
    C --> E["OpenAiCompatibleProvider(...)"]
    C --> F["FailingMockLlmProvider()"]
    C --> G["PartialFailingMockLlmProvider()"]
    C --> H["FlakyMockLlmProvider()"]
    C --> I["TimeoutMockLlmProvider()"]

    B --> J["ReliabilityPolicy(...)"]
    B --> K["ReliableLlmProvider(base_provider, policy)"]
```

`ChatService` never talks directly to a concrete provider class. It always receives the reliability wrapper built by `build_provider()`.

## 8. Repository Call Map

```mermaid
flowchart LR
    UserService --> SyncUserDatabase
    SyncUserDatabase --> UserRepository
    UserService --> RefreshTokenRepository

    ChatService --> ConversationRepository
    ChatService --> MessageRepository

    UserRepository --> UserModel
    RefreshTokenRepository --> RefreshTokenModel
    ConversationRepository --> ConversationModel
    MessageRepository --> MessageModel
```

At the class level, services orchestrate work and repositories handle database persistence and loading.

## 9. Middleware And Error Handling Timing

```mermaid
flowchart TD
    A["HTTP request arrives"] --> B["log_requests middleware"]
    B --> C{"content-length too large?"}
    C -- yes --> D["JSONResponse 413"]
    C -- no --> E["route dependencies and handler"]
    E --> F{"AppError raised?"}
    F -- yes --> G["register_exception_handlers() mapping"]
    F -- no --> H["normal response"]
    G --> I["JSON error response"]
    H --> J["middleware logs duration and status"]
    I --> J
```

Middleware runs before and after the route handler. Exception handlers only take over if a route or service raises an application error.
