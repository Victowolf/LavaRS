import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL = "microsoft/phi-3.5-vision-instruct"

print("=== Starting RS-LLaVA FastAPI server ===")
print("=== Loading Phi-3.5 Vision Instruct ===")

device = "cuda" if torch.cuda.is_available() else "cpu"

# IMPORTANT: trust_remote_code=True
processor = AutoProcessor.from_pretrained(
    MODEL,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

app = FastAPI()


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
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

    result = processor.decode(output[0], skip_special_tokens=True)
    return JSONResponse({"response": result})
