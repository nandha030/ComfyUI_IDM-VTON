"""
Run this script ONCE on a RunPod Network Volume to pre-download IDM-VTON weights.

Usage:
    python download_models.py [--output-dir /runpod-volume/IDM-VTON]

On RunPod:
1. Spin up a GPU pod with your network volume attached.
2. SSH in and run:  python download_models.py
3. Terminate the pod — weights stay on the volume.
4. All serverless endpoints using the same volume will skip the download.
"""

import os
import argparse
from huggingface_hub import snapshot_download

HF_REPO_ID = "yisol/IDM-VTON"
DEFAULT_OUTPUT = os.environ.get("IDM_VTON_WEIGHTS_PATH", "/runpod-volume/IDM-VTON")


def main():
    parser = argparse.ArgumentParser(description="Download IDM-VTON weights to a directory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Destination directory")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HuggingFace token (if repo is gated)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Downloading {HF_REPO_ID} → {args.output_dir} ...")

    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
        token=args.hf_token,
    )

    print("Download complete.")
    print(f"Set IDM_VTON_WEIGHTS_PATH={args.output_dir} in your RunPod endpoint environment.")


if __name__ == "__main__":
    main()
