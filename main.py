# main.py
import os
import logging
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# Try to import BitsAndBytes config (newer transformers expose it), fall back gracefully
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

MODEL_NAME = "liuhaotian/llava-v1.5-7b"  # public LLaVA-style model (change if you prefer another public model)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lava-server")

app = FastAPI(title="RS-LLaVA 4bit Server")

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Device: {device}")

# Quantization / bitsandbytes config
if BitsAndBytesConfig is not None:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
else:
    # If BitsAndBytesConfig isn't available in transformers version, we'll pass legacy args below.
    bnb_config = None

log.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

log.info("Loading model in 4-bit (bitsandbytes)...")
load_kwargs = dict(
    trust_remote_code=True,
    device_map="auto",
)

if bnb_config is not None:
    load_kwargs["quantization_config"] = bnb_config
else:
    # legacy fallback: transformers may accept load_in_4bit directly
    load_kwargs["load_in_4bit"] = True
    load_kwargs["bnb_4bit_use_double_quant"] = True
    load_kwargs["bnb_4bit_quant_type"] = "nf4"
    # compute dtype
    load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
model.eval()

log.info("Model loaded ✅")

# Generation config default
default_gen_cfg = GenerationConfig(
    max_new_tokens=256,
    temperature=0.2,
    top_p=0.9,
    do_sample=True,
)

@app.post("/generate")
async def generate(prompt: str = Form(...)):
    """
    POST /generate
    form-data:
      - prompt: string
    returns JSON { "response": "<generated text>" }
    """
    if not prompt:
        return JSONResponse({"error": "prompt required"}, status_code=400)

    inputs = tokenizer(prompt, return_tensors="pt")
    # Move inputs to model device(s)
    try:
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    except StopIteration:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            generation_config=default_gen_cfg,
        )

    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return {"response": text}


@app.get("/")
def root():
    return {"status": "RS-LLaVA 4bit server running"}
