"""
RunPod Serverless Handler for IDM-VTON
Accepts base64-encoded images, returns base64-encoded result.
"""

import os
import sys
import base64
import io
import torch
import runpod
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModelWithProjection,
    CLIPTextModel,
)
from huggingface_hub import snapshot_download

# Add src to path so we can import the pipeline without ComfyUI
SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, SRC_PATH)

from idm_vton.unet_hacked_tryon import UNet2DConditionModel
from idm_vton.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from idm_vton.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

# ── Config ───────────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.environ.get("IDM_VTON_WEIGHTS_PATH", "/runpod-volume/IDM-VTON")
HF_REPO_ID   = "yisol/IDM-VTON"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global pipeline (lazy-loaded on first request) ───────────────────────────
PIPELINE = None


def ensure_weights():
    """Download weights from HuggingFace if not already present."""
    scheduler_dir = os.path.join(WEIGHTS_PATH, "scheduler")
    if not os.path.isdir(scheduler_dir):
        print(f"[IDM-VTON] Weights not found at {WEIGHTS_PATH}. Downloading from HuggingFace...")
        os.makedirs(WEIGHTS_PATH, exist_ok=True)
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=WEIGHTS_PATH,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
        )
        print("[IDM-VTON] Download complete.")
    else:
        print(f"[IDM-VTON] Weights found at {WEIGHTS_PATH}.")


def load_pipeline(weight_dtype_str: str = "float16"):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    weight_dtype = dtype_map.get(weight_dtype_str, torch.float16)

    ensure_weights()
    print(f"[IDM-VTON] Loading pipeline ({weight_dtype_str}) ...")

    noise_scheduler = DDPMScheduler.from_pretrained(WEIGHTS_PATH, subfolder="scheduler")

    vae = (
        AutoencoderKL.from_pretrained(WEIGHTS_PATH, subfolder="vae", torch_dtype=weight_dtype)
        .requires_grad_(False).eval().to(DEVICE)
    )
    unet = (
        UNet2DConditionModel.from_pretrained(WEIGHTS_PATH, subfolder="unet", torch_dtype=weight_dtype)
        .requires_grad_(False).eval().to(DEVICE)
    )
    image_encoder = (
        CLIPVisionModelWithProjection.from_pretrained(WEIGHTS_PATH, subfolder="image_encoder", torch_dtype=weight_dtype)
        .requires_grad_(False).eval().to(DEVICE)
    )
    unet_encoder = (
        UNet2DConditionModel_ref.from_pretrained(WEIGHTS_PATH, subfolder="unet_encoder", torch_dtype=weight_dtype)
        .requires_grad_(False).eval().to(DEVICE)
    )
    text_encoder_one = (
        CLIPTextModel.from_pretrained(WEIGHTS_PATH, subfolder="text_encoder", torch_dtype=weight_dtype)
        .requires_grad_(False).eval().to(DEVICE)
    )
    text_encoder_two = (
        CLIPTextModelWithProjection.from_pretrained(WEIGHTS_PATH, subfolder="text_encoder_2", torch_dtype=weight_dtype)
        .requires_grad_(False).eval().to(DEVICE)
    )
    tokenizer_one = AutoTokenizer.from_pretrained(WEIGHTS_PATH, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(WEIGHTS_PATH, subfolder="tokenizer_2", use_fast=False)

    pipe = TryonPipeline.from_pretrained(
        WEIGHTS_PATH,
        unet=unet,
        vae=vae,
        feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        scheduler=noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=weight_dtype,
    )
    pipe.unet_encoder = unet_encoder
    pipe = pipe.to(DEVICE)
    pipe.weight_dtype = weight_dtype

    print("[IDM-VTON] Pipeline ready.")
    return pipe


def decode_image(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_inference(pipe, job_input: dict) -> dict:
    width               = int(job_input.get("width", 768))
    height              = int(job_input.get("height", 1024))
    num_inference_steps = int(job_input.get("num_inference_steps", 30))
    guidance_scale      = float(job_input.get("guidance_scale", 2.0))
    strength            = float(job_input.get("strength", 1.0))
    seed                = int(job_input.get("seed", 42))
    garment_description = job_input.get("garment_description", "")
    negative_prompt     = job_input.get("negative_prompt", "monochrome, lowres, bad anatomy, worst quality, low quality")

    human_img   = decode_image(job_input["human_img"]).resize((width, height))
    garment_img = decode_image(job_input["garment_img"]).resize((width, height))
    pose_img    = decode_image(job_input["pose_img"]).resize((width, height))
    mask_img    = decode_image(job_input["mask_img"]).resize((width, height))

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.inference_mode():
                prompt = "model is wearing " + garment_description
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

                prompt_c = ["a photo of " + garment_description]
                (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                    prompt_c,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=[negative_prompt],
                )

                pose_tensor    = tensor_transform(pose_img).unsqueeze(0).to(DEVICE, pipe.dtype)
                garment_tensor = tensor_transform(garment_img).unsqueeze(0).to(DEVICE, pipe.dtype)

                output_images = pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator(DEVICE).manual_seed(seed),
                    strength=strength,
                    pose_img=pose_tensor,
                    text_embeds_cloth=prompt_embeds_c,
                    cloth=garment_tensor,
                    mask_image=mask_img,
                    image=human_img,
                    height=height,
                    width=width,
                    ip_adapter_image=garment_img,
                    guidance_scale=guidance_scale,
                )[0]

    result_b64 = [encode_image(img) for img in output_images]
    return {"images": result_b64}


def handler(job):
    global PIPELINE

    job_input = job.get("input", {})

    required = ["human_img", "garment_img", "pose_img", "mask_img", "garment_description"]
    missing = [k for k in required if k not in job_input]
    if missing:
        return {"error": f"Missing required fields: {missing}"}

    try:
        if PIPELINE is None:
            weight_dtype = job_input.get("weight_dtype", os.environ.get("WEIGHT_DTYPE", "float16"))
            PIPELINE = load_pipeline(weight_dtype)

        return run_inference(PIPELINE, job_input)

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
