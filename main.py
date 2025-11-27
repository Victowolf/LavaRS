import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    AutoTokenizer
)
from peft import PeftModel
from PIL import Image
from io import BytesIO

app = FastAPI(title="RS-LLaVA-LoRA Server")

# ------------------------------
# Model identifiers
# ------------------------------
BASE_MODEL = "liuhaotian/llava-v1.5-7b"
LORA_ADAPTER = "BigData-KSU/RS-llava-v1.5-7b-LoRA"

print("🚀 Loading RS-LLaVA model...")

# ------------------------------
# Load processor & tokenizer
# ------------------------------
processor = LlavaProcessor.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# ------------------------------
# Load base model (FP16)
# ------------------------------
base_model = LlavaForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ------------------------------
# Apply LoRA
# ------------------------------
model = PeftModel.from_pretrained(
    base_model,
    LORA_ADAPTER,
    torch_dtype=torch.float16
)

model.eval()
print("✅ RS-LLaVA + LoRA loaded successfully!")


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    image: UploadFile = File(None)
):
    """
    RS-LLaVA multimodal inference endpoint.
    """

    pil_image = None

    if image:
        pil_image = Image.open(BytesIO(await image.read())).convert("RGB")

    # Process image + text
    if pil_image:
        inputs = processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(model.device)
    else:
        inputs = processor(
            text=prompt,
            return_tensors="pt"
        ).to(model.device)

    # Generate output
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return JSONResponse({"response": response})


@app.get("/")
def home():
    return {"status": "RS-LLaVA LoRA server running"}
