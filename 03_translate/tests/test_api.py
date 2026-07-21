from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api
from api import (
  TranslationDirectionEnum,
  TranslationRequest,
  TranslationResponse,
)


@pytest.fixture
def client():
  tokenizer = MagicMock()
  model = MagicMock()

  encoded = MagicMock()
  encoded.to.return_value = encoded

  tokenizer.return_value = encoded
  tokenizer.batch_decode.return_value = [
    "Hello",
    "World",
  ]

  model.generate.return_value = ["fake_tokens"]

  api.app.state[TranslationDirectionEnum.EN_TO_UK] = (
    tokenizer,
    model,
  )

  api.app.state[TranslationDirectionEnum.UK_TO_EN] = (
    tokenizer,
    model,
  )

  return TestClient(api.app)


def test_health(client):
  response = client.get("/health")

  assert response.status_code == 200

  body = response.json()

  assert body["status"] == "ok"
  assert body["device"] == api.DEVICE
  assert body["model_en_to_uk"] == api.MODEL_EN_TO_UK
  assert body["model_uk_to_en"] == api.MODEL_UK_TO_EN


def test_translate_en_to_uk(client):
  response = client.post(
    "/translate",
    json={
      "texts": ["Hello", "World"],
      "direction": "en_to_uk",
    },
  )

  assert response.status_code == 200
  assert response.json() == {
    "translations": [
      "Hello",
      "World",
    ]
  }


def test_translate_uk_to_en(client):
  response = client.post(
    "/translate",
    json={
      "texts": ["Привіт"],
      "direction": "uk_to_en",
    },
  )

  assert response.status_code == 200
  assert response.json()["translations"] == [
    "Hello",
    "World",
  ]


def test_translate_calls_tokenizer_and_model(client):
  tokenizer, model = api.app.state[
    TranslationDirectionEnum.EN_TO_UK
  ]

  response = client.post(
    "/translate",
    json={
      "texts": ["Hello"],
      "direction": "en_to_uk",
    },
  )

  assert response.status_code == 200

  tokenizer.assert_called_once_with(
    ["Hello"],
    return_tensors="pt",
    padding=True,
    truncation=True,
  )

  model.generate.assert_called_once()

  tokenizer.batch_decode.assert_called_once_with(
    model.generate.return_value,
    skip_special_tokens=True,
  )


def test_invalid_direction(client):
  response = client.post(
    "/translate",
    json={
      "texts": ["Hello"],
      "direction": "bad_direction",
    },
  )

  assert response.status_code == 422


def test_request_models():
  req = TranslationRequest(
    texts=["Hello"],
  )

  assert req.direction == TranslationDirectionEnum.EN_TO_UK

  resp = TranslationResponse(
    translations=["Привіт"],
  )

  assert resp.translations == ["Привіт"]
