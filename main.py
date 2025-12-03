import os
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig
from fastapi.middleware.cors import CORSMiddleware


# Disable FlashAttention
os.environ["FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"
os.environ["FLASHATTENTION_DISABLED"] = "1"
os.environ["HF_DISABLE_FLASH_ATTENTION"] = "1"

MODEL = "microsoft/phi-3.5-vision-instruct"

print("=== Starting RS-LLaVA FastAPI server ===")
print("=== Loading Phi-3.5 Vision Instruct ===")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load config (force eager attention)
config = AutoConfig.from_pretrained(
    MODEL,
    trust_remote_code=True
)
config._attn_implementation = "eager"
config.attn_implementation = "eager"

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    config=config,
)

# Load processor
processor = AutoProcessor.from_pretrained(
    MODEL,
    trust_remote_code=True,
    num_crops=1,   # recommended for single image
)

app = FastAPI()

# -------------------------
# ✅ Enable CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Change to specific domains for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    # Load image
    img = Image.open(image.file).convert("RGB")

    # Build messages according to official format
    messages = [
        {
            "role": "user",
            "content": f"<|image_1|>\n{prompt}"
        }
    ]

    # Convert to model-ready text
    prompt_text = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Processor expects a list of images
    inputs = processor(
        prompt_text,
        [img],
        return_tensors="pt"
    ).to(device)

    # Generate
    output = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.0,
        do_sample=False,
        use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id
    )

    # Remove prompt tokens
    output_ids = output[:, inputs["input_ids"].shape[1]:]

    result = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return JSONResponse({"response": result})


@app.get("/")
def root():
    return {"status": "phi-3.5 vision is live"}
