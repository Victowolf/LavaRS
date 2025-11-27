import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPProcessor,
    CLIPModel
)

MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

app = FastAPI(title="LLaVA Server")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=== Loading LLaVA 1.5 7B (public) ===")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# vision + text get loaded separately
vision_tower = CLIPModel.from_pretrained(
    MODEL_NAME,
    subfolder="vision_tower"
).to(device)

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME,
    subfolder="vision_tower"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)

print("=== Model Loaded Successfully ===")

@app.post("/generate")
async def generate(prompt: str = Form(...), image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")

    vision_inputs = processor(images=img, return_tensors="pt").to(device)
    text_inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **text_inputs,
        max_new_tokens=200,
        temperature=0.2
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": result}


@app.get("/")
def home():
    return {"status": "llava server running"}
