# RunPod Serverless – IDM-VTON
# Base image: RunPod PyTorch 2.2.0 + CUDA 12.1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements_runpod.txt .
RUN pip install --no-cache-dir -r requirements_runpod.txt

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . .

# ── Environment ───────────────────────────────────────────────────────────────
# Override with your Network Volume mount path if different
ENV IDM_VTON_WEIGHTS_PATH=/runpod-volume/IDM-VTON
ENV WEIGHT_DTYPE=float16
# Disable HuggingFace symlinks (not supported on all network volumes)
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV PYTHONUNBUFFERED=1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
