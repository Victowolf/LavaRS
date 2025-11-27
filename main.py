import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import LlavaProcessor, LlavaForConditionalGeneration

MODEL_NAME = "liuhaotian/llava-v1.5-7b"

app = FastAPI(title="RS-LLaVA Server")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=== Loading RS-LLaVA 7B (public model) ===")

processor = LlavaProcessor.from_pretrained(MODEL_NAME)
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)

print("=== Model loaded successfully ===")

@app.post("/generate")
async def generate(prompt: str = Form(...), image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")

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
