FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md alembic.ini ./
COPY migrations ./migrations
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install .

EXPOSE 8000

CMD ["uvicorn", "ai_chat.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
