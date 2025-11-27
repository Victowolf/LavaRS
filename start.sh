#!/usr/bin/env bash
set -euo pipefail

echo "=== start.sh: create venv and install requirements (if not done) ==="

# create venv if not present
if [ ! -d "venv" ]; then
  python -m venv venv
fi

# activate
source venv/bin/activate

# pip upgrade
pip install --upgrade pip

echo "Installing python requirements (may take a while)..."
pip install -r requirements.txt

# optional: pre-warm HF cache or pre-download models if you want to avoid download at runtime
# Uncomment to pre-download:
# python - <<PY
# from transformers import AutoTokenizer, AutoModelForCausalLM
# AutoTokenizer.from_pretrained("llava/llava-1.5-7b-hf", trust_remote_code=True)
# AutoModelForCausalLM.from_pretrained("llava/llava-1.5-7b-hf", trust_remote_code=True)
# PY

echo "Starting uvicorn..."
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
