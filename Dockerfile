# RunPod Serverless – IDM-VTON
# Base image already includes: Python 3.10, PyTorch 2.2.0, CUDA 12.1, torchvision, torchaudio
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (torch/torchvision/torchaudio already in base image) ─
COPY requirements_runpod.txt .
RUN pip install --no-cache-dir \
        runpod>=1.6.0 \
        accelerate==0.30.0 \
        transformers==4.40.2 \
        diffusers==0.27.2 \
        huggingface_hub>=0.22.0 \
        einops==0.8.0 \
        bitsandbytes==0.43.0 \
        scipy==1.13.0 \
        torchmetrics==1.4.0 \
        tqdm==4.66.4 \
        Pillow>=10.0.0 \
        opencv-python-headless

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . .

# ── Environment ───────────────────────────────────────────────────────────────
ENV IDM_VTON_WEIGHTS_PATH=/runpod-volume/IDM-VTON
ENV WEIGHT_DTYPE=float16
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV PYTHONUNBUFFERED=1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
