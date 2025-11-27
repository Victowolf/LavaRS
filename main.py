import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
)
from peft import PeftModel
from PIL import Image
from io import BytesIO

app = FastAPI(title="RS-LLaVA FastAPI Server")

BASE_MODEL = "llava-hf/llava-1.5-7b-hf"
LORA_ADAPTER = "BigData-KSU/RS-llava-v1.5-7b-LoRA"

print("=== Loading RS-LLaVA 7B ===")

# FIX: Disable fast tokenizer (LLaVA incompatible with fast version)
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    use_fast=False
)

processor = AutoProcessor.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# load LoRA
model = PeftModel.from_pretrained(model, LORA_ADAPTER)
model = model.merge_and_unload()
model.eval()

print("✔ RS-LLaVA Model Ready!")


@app.post("/generate")
async def generate(prompt: str = Form(...), image: UploadFile = File(None)):

    pil_img = None
    if image:
        pil_img = Image.open(BytesIO(await image.read())).convert("RGB")

    inputs = processor(
        text=prompt,
        images=pil_img,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": result}


@app.get("/")
def home():
    return {"status": "RS-LLaVA server running"}
