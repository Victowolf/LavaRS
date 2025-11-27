import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor

MODEL = "microsoft/phi-3.5-vision-instruct"

app = FastAPI()

print("=== Loading Phi-3.5 Vision Instruct ===")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("=== Model Loaded ===")

@app.post("/generate")
async def generate(prompt: str = Form(...), image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")

    inputs = processor(
        text=prompt,
        images=img,
        return_tensors="pt"
    ).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.2
    )

    response = processor.decode(output[0], skip_special_tokens=True)
    return JSONResponse({"response": response})
