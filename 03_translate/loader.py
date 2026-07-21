import asyncio

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def loader(model: str):
  print(f"Loading {model}...")

  AutoTokenizer.from_pretrained(model)
  AutoModelForSeq2SeqLM.from_pretrained(model)

  print("Model ready.")

async def async_loader(model_name: str):
    return await asyncio.to_thread(loader, model_name)
