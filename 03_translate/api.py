from contextlib import asynccontextmanager
from enum import Enum

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import MODEL_EN_TO_UK, MODEL_UK_TO_EN

DEVICE = (
  "mps"
  if torch.backends.mps.is_available()
  else "cuda"
  if torch.cuda.is_available()
  else "cpu"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
  app.state[TranslationDirectionEnum.EN_TO_UK] = (
    AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-uk"),
    AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-uk").to(DEVICE),
  )

  app.state[TranslationDirectionEnum.UK_TO_EN] = (
    AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-uk-en"),
    AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-uk-en").to(DEVICE),
  )

  yield


app = FastAPI(lifespan=lifespan)

class TranslationDirectionEnum(str, Enum):
    EN_TO_UK = "en_to_uk"
    UK_TO_EN = "uk_to_en"

class TranslationRequest(BaseModel):
  texts: list[str]
  direction: TranslationDirectionEnum = TranslationDirectionEnum.EN_TO_UK


class TranslationResponse(BaseModel):
  translations: list[str]

@app.get("/health")
def health():
  return {
    "status": "ok",
    "device": DEVICE,
    "model_uk_to_en": MODEL_UK_TO_EN,
    "model_en_to_uk": MODEL_EN_TO_UK,
  }


@app.post("/translate", response_model=TranslationResponse)
def translate(req: TranslationRequest):

  tokenizer, model = app.state[req.direction]

  inputs = tokenizer(
    req.texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
  ).to(DEVICE)

  with torch.inference_mode():
    generated = model.generate(
      **inputs,
      max_new_tokens=512,
    )

  translations = tokenizer.batch_decode(
    generated,
    skip_special_tokens=True,
  )

  return TranslationResponse(translations=translations)
