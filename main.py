import os
import torch
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# Disable FlashAttention (stable on MIG / shared GPU)
# --------------------------------------------------
os.environ["HF_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["FLASH_ATTENTION"] = "0"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"

MODEL = "microsoft/phi-3.5-vision-instruct"

print("=== Starting Phi-3.5 Text Server ===")
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True
)

tokenizer.model_max_length = 4096

# --------------------------------------------------
# Load model
# --------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
)

model.eval()

# --------------------------------------------------
# FastAPI setup
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Text endpoint (renamed)
# --------------------------------------------------
@app.post("/askgemini")
async def askgemini(prompt: str = Form(...)):

    messages = [
        {"role": "user", "content": prompt}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id
    )

    output_ids = output[:, inputs["input_ids"].shape[1]:]

    result = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return JSONResponse({"response": result})


# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "phi-3.5 text server live"}
