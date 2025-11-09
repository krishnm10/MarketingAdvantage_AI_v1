import os
import requests
import base64
from datetime import datetime
from typing import Dict, Any

# Optional local diffusers fallback
try:
    from diffusers import StableDiffusionPipeline
    import torch
    _HAS_DIFFUSERS = True
except Exception:
    _HAS_DIFFUSERS = False

# === Config ===
AI_IMAGE_MODEL = os.getenv("AI_IMAGE_MODEL", "deepai").lower()  # deepai | dalle | diffusers_local
DEEPAI_API_KEY = os.getenv("DEEPAI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OUTPUT_DIR = os.getenv("IMAGE_OUTPUT_DIR", "./generated_images")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================================
# 🔹 DeepAI Generator
# ==========================================================
def _generate_with_deepai(prompt: str) -> Dict[str, Any]:
    """Generate image using DeepAI Text-to-Image API."""
    try:
        print(f"[DeepAI] Generating image for prompt: {prompt[:100]}...")
        response = requests.post(
            "https://api.deepai.org/api/text2img",
            data={"text": prompt},
            headers={"api-key": DEEPAI_API_KEY},
            timeout=60
        )
        response.raise_for_status()

        data = response.json()
        image_url = data.get("output_url")

        if not image_url:
            raise ValueError("No image URL returned from DeepAI")

        filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_deepai.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        img_data = requests.get(image_url).content
        with open(filepath, "wb") as f:
            f.write(img_data)

        return {
            "status": "success",
            "model_used": "DeepAI Text-to-Image",
            "file_path": filepath,
            "image_url": image_url,
            "prompt_used": prompt
        }

    except Exception as e:
        return {
            "status": "error",
            "model_used": "DeepAI Text-to-Image",
            "error": str(e),
            "prompt_used": prompt
        }


# ==========================================================
# 🔹 DALL·E (OpenAI)
# ==========================================================
def _generate_with_dalle(prompt: str) -> Dict[str, Any]:
    """Generate an image using OpenAI’s DALL·E 3 API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        result = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024"
        )

        image_url = result.data[0].url
        filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_dalle.png"
        filepath = os.path.join(OUTPUT_DIR, filename)

        img_data = requests.get(image_url).content
        with open(filepath, "wb") as f:
            f.write(img_data)

        return {
            "status": "success",
            "model_used": "DALL·E 3",
            "file_path": filepath,
            "image_url": image_url,
            "prompt_used": prompt
        }

    except Exception as e:
        return {
            "status": "error",
            "model_used": "DALL·E 3",
            "error": str(e),
            "prompt_used": prompt
        }


# ==========================================================
# 🔹 Local Diffusers Fallback
# ==========================================================
def _generate_with_diffusers(prompt: str) -> Dict[str, Any]:
    """Generate image locally using Hugging Face diffusers."""
    if not _HAS_DIFFUSERS:
        return {"status": "error", "error": "Diffusers not installed", "prompt_used": prompt}

    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        image = pipe(prompt).images[0]
        filename = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_diffusers.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)

        return {
            "status": "success",
            "model_used": "Local Stable Diffusion (Diffusers)",
            "file_path": filepath,
            "prompt_used": prompt
        }

    except Exception as e:
        return {
            "status": "error",
            "model_used": "Diffusers Local",
            "error": str(e),
            "prompt_used": prompt
        }


# ==========================================================
# 🔹 Unified Entry Point (Conditional Trigger)
# ==========================================================
def generate_image_from_brief(prompt: str, visual_request: bool = False) -> Dict[str, Any]:
    """
    Generate image only when explicitly requested (visual/video context).
    Selects DeepAI → DALL·E → Diffusers fallback automatically.
    """
    if not visual_request:
        print("[ImageGen] Skipped — user did not request visual output.")
        return {
            "status": "skipped",
            "reason": "Visual generation not requested",
            "prompt_used": prompt
        }

    if AI_IMAGE_MODEL == "deepai" and DEEPAI_API_KEY:
        return _generate_with_deepai(prompt)
    elif AI_IMAGE_MODEL == "dalle" and OPENAI_API_KEY:
        return _generate_with_dalle(prompt)
    elif AI_IMAGE_MODEL == "diffusers_local":
        return _generate_with_diffusers(prompt)
    else:
        return {
            "status": "error",
            "error": f"No valid image model configured or missing API key ({AI_IMAGE_MODEL})",
            "prompt_used": prompt
        }
