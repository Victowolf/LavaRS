import torch
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration

MODEL_NAME = "HuggingFaceM4/llava-v1.5-7b-hf"

app = FastAPI(title="RS-LLaVA Server")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=== Loading RS-LLaVA 7B ===")

processor = LlavaProcessor.from_pretrained(MODEL_NAME)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)

print("=== Model loaded ===")

@app.post("/generate")
async def generate(prompt: str = Form(...), image: UploadFile = File(None)):
    if image:
        img = Image.open(image.file).convert("RGB")
    else:
        return JSONResponse({"error": "Image required"}, status_code=400)

    inputs = processor(prompt, img, return_tensors="pt").to(device, torch.float16)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2
    )

    text = processor.decode(output[0], skip_special_tokens=True)
    return {"response": text}


@app.get("/")
def home():
    return {"status": "RS-LLaVA server running"}
