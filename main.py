# main.py
import os
import json
import torch
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

# Transformers + PEFT
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

# Prevent accidental use of FlashAttention if you prefer eager attention
os.environ.setdefault("DISABLE_FLASH_ATTENTION", "1")
os.environ.setdefault("FLASH_ATTENTION", "0")

app = FastAPI(title="RS-LLaVA (LLaVA+LoRA) server")

# --- Config: change these to your preferred base / lora
BASE_MODEL = "llava/llava-1.5-7b-hf"                # base LLaVA (HF-style)
LORA_WEIGHTS = "BigData-KSU/RS-llava-v1.5-7b-LoRA"  # LoRA weights on HF hub

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading tokenizer from {BASE_MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print(f"Loading base model from {BASE_MODEL} (this may take a while) ...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
)

print(f"Attaching LoRA weights from {LORA_WEIGHTS} ...")
# Load LoRA adapter on top of base model
model = PeftModel.from_pretrained(
    base_model,
    LORA_WEIGHTS,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    # set `is_trainable=False` if you only want to inference (default behaviour)
)

model.eval()

# Helpful generation config (tweak to taste)
gen_config = GenerationConfig(
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.9,
)

class TextRequest(BaseModel):
    prompt: str
    # image is optional, but currently the example below treats images as "not-yet-processed"
    # Full image support requires access to the model's image processor/processor code from the repo.
    image: Optional[str] = None  # base64 or url — placeholder

@app.post("/generate")
async def generate(request: TextRequest):
    """
    Text-only endpoint. For a simple test:
    POST JSON {"prompt": "Explain Newton's laws in simple words"}
    """
    prompt = request.prompt

    # NOTE: LLaVA/RS-LLaVA may require special chat templating or chat wrappers.
    # Here we do a simple tokenization/generate path that works for many HF causal LM wrappers.
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_config,
            do_sample=True,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return JSONResponse({"response": text})


@app.post("/generate_with_image")
async def generate_with_image(prompt: str = Form(...), image: UploadFile = File(...)):
    """
    This route accepts an image; currently it returns guidance on how to enable true multimodal inference.
    Implementing full image-aware chat needs the model's image-processor (often in the repo),
    plus the model-specific `chat`/`generate` wrapper which expects pixel-values or preprocessed tensors.

    To enable multimodal:
      1. Load the repo's processor (e.g. a VisionProcessor / InternVLImageProcessor / LLaVAProcessor).
      2. Convert image to pixel-values / tensor as the repo expects.
      3. Call the model.chat(...) or model.generate(...) using the repo's recommended signature.

    If you want, I can modify this endpoint to call the repo-specific processor once you confirm
    the repo layout / processor class name (e.g. `LavaProcessor`, `LLaVAProcessor`, etc).
    """
    return JSONResponse({
        "error": "image endpoint not enabled yet",
        "hint": "See README in the cloned repo for the model processor and usage pattern (processor(images=...)."
    }, status_code=501)


@app.get("/")
def status():
    return {"status": "RS-LLaVA server (text-only) ready"}
