from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import MODEL

DEVICE = (
  "mps"
  if torch.backends.mps.is_available()
  else "cuda"
  if torch.cuda.is_available()
  else "cpu"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
  tokenizer = AutoTokenizer.from_pretrained(MODEL)
  model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(DEVICE)

  app.state.tokenizer = tokenizer
  app.state.model = model

  yield


app = FastAPI(lifespan=lifespan)


class TranslationRequest(BaseModel):
  texts: list[str]
  src_lang: str = "eng_Latn"
  tgt_lang: str = "ukr_Cyrl"


class TranslationResponse(BaseModel):
  translations: list[str]

@app.get("/health")
def health():
  return {
    "status": "ok",
    "device": DEVICE,
    "model": MODEL,
  }


@app.post("/translate", response_model=TranslationResponse)
def translate(req: TranslationRequest):

  tokenizer = app.state.tokenizer
  model = app.state.model

  tokenizer.src_lang = req.src_lang

  inputs = tokenizer(
    req.texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
  ).to(DEVICE)

  with torch.inference_mode():
    generated = model.generate(
      **inputs,
      forced_bos_token_id=tokenizer.convert_tokens_to_ids(
        req.tgt_lang
      ),
      max_length=512,
    )

  translations = tokenizer.batch_decode(
    generated,
    skip_special_tokens=True,
  )

  return TranslationResponse(translations=translations)