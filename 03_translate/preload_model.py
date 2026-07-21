from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "facebook/nllb-200-distilled-600M"

print(f"Loading {MODEL}...")

AutoTokenizer.from_pretrained(MODEL)
AutoModelForSeq2SeqLM.from_pretrained(MODEL)

print("Model ready.")