import os
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

# Force-disable FlashAttention globally
os.environ["FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["FLASHATTENTION_DISABLED"] = "1"
os.environ["HF_DISABLE_FLASH_ATTENTION"] = "1"

MODEL = "microsoft/phi-3.5-vision-instruct"

print("=== Starting RS-LLaVA FastAPI server ===")
print("=== Loading Phi-3.5 Vision Instruct ===")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load config first and override attn settings
config = AutoConfig.from_pretrained(
    MODEL,
    trust_remote_code=True
)

# IMPORTANT — hard-disable flash-attn at the config level
config.attn_implementation = "eager"
config._attn_implementation_internal = "eager"
config.use_flash_attention = False
config.flash_attn = False
config.enable_flash_attention = False

# Load processor
processor = AutoProcessor.from_pretrained(
    MODEL,
    trust_remote_code=True
)

# Load model with patched config
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    config=config,              # 👈 Force our patched config
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

app = FastAPI()


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    img = Image.open(image.file).convert("RGB")

    # Phi-3.5 requires <image> tag for each image
    if "<image>" not in prompt:
        prompt = "<image>\n" + prompt

    # Run processor
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

    # Decode
    result = processor.decode(output[0], skip_special_tokens=True)
    return JSONResponse({"response": result})


@app.get("/")
def root():
    return {"status": "phi-3.5 vision is live"}
