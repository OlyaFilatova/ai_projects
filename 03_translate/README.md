## setup environment

```
python3 -m venv .venv
source .venv/bin/activate
```

## install requirements

`pip install -r requirements.txt`

## run in Docker

`docker compose up --build`

## run locally

`python3 preload_model.py`

`uvicorn api:app --host 0.0.0.0 --port 8000`

## call API

health check

`curl http://localhost:8000/health`

english to ukrainian translation

```
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Hello!",
      "How are you?",
      "Artificial intelligence is changing translation."
    ],
    "direction": "en_to_uk"
  }'
```

ukrainian to english translation

```
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Привіт!",
      "Як справи?",
      "Штучний інтелект швидко розвивається."
    ],
    "direction": "uk_to_en"
  }'
```

