"""
Module 3A: Asset Preprocessor
Path: src/generation/preprocessor.py

Handles raw client image ingestion, background extraction via rembg,
and 4x super-resolution upscaling via Replicate Real-ESRGAN.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Union
import httpx

# Configure module logger
logger = logging.getLogger("generation.preprocessor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class AssetPreprocessor:
    """
    Ingests raw client product assets, isolates foreground product subjects,
    and scales assets to high-resolution staging assets for generative video models.
    """

    REPLICATE_MODEL_VERSION = (
        "nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b"
    )

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("storage/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def remove_background(self, input_path: Union[str, Path], output_png: Union[str, Path]) -> Path:
        """
        Removes background using rembg (u2net model).
        
        WARNING: First execution downloads the ~170MB u2net model to ~/.u2net/u2net.onnx.
        Post-install warmup command:
            python -c 'from rembg import remove'

        Error handling:
            On rembg failure/crash, catches Exception, logs WARNING,
            and outputs the original image bytes as fallback.
        """
        input_path = Path(input_path)
        output_png = Path(output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)

        try:
            from rembg import remove

            with open(input_path, 'rb') as inp:
                output_bytes = remove(inp.read())
            
            with open(output_png, 'wb') as out:
                out.write(output_bytes)
            logger.info("rembg successfully isolated subject: %s -> %s", input_path, output_png)
            return output_png

        except Exception as exc:
            logger.warning("rembg failed: %s. Falling back to original image bytes.", exc)
            with open(input_path, 'rb') as inp:
                fallback_bytes = inp.read()
            with open(output_png, 'wb') as out:
                out.write(fallback_bytes)
            return output_png

    def upscale_real_esrgan(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        scale: int = 4,
        face_enhance: bool = False
    ) -> Path:
        """
        Upscales an image using Replicate Real-ESRGAN.

        Error handling:
            If Replicate fails, retries once after a 2-second sleep.
            If the retry fails, falls back to the original cleaned image without upscaling.
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                import replicate

                logger.info(
                    "Executing Replicate Real-ESRGAN (attempt %d/%d) on %s",
                    attempt,
                    max_attempts,
                    image_path
                )
                with open(image_path, 'rb') as img_file:
                    output = replicate.run(
                        self.REPLICATE_MODEL_VERSION,
                        input={'image': img_file, 'scale': scale, 'face_enhance': face_enhance}
                    )

                # Replicate returns a URL (string) or list of URLs to the generated image
                target_url = output if isinstance(output, str) else output[0]
                
                # Download output URL to local path
                response = httpx.get(target_url, timeout=60.0)
                response.raise_for_status()

                with open(output_path, 'wb') as out_file:
                    out_file.write(response.content)

                logger.info("Real-ESRGAN upscale successful: %s", output_path)
                return output_path

            except Exception as exc:
                logger.warning(
                    "Replicate Real-ESRGAN attempt %d failed: %s",
                    attempt,
                    exc
                )
                if attempt < max_attempts:
                    logger.info("Retrying Replicate call in 2 seconds...")
                    time.sleep(2)
                else:
                    logger.warning(
                        "Replicate Real-ESRGAN failed after %d attempts. "
                        "Falling back to original cleaned image without upscaling.",
                        max_attempts
                    )
                    with open(image_path, 'rb') as src, open(output_path, 'wb') as dst:
                        dst.write(src.read())
                    return output_path

    def process(
        self,
        raw_asset_path: Union[str, Path],
        asset_id: str,
        remove_bg: bool = True,
        upscale: bool = True
    ) -> Path:
        """
        Full 2-stage preprocessing pipeline:
        1. Background removal -> {asset_id}_cleaned.png
        2. Super-resolution 4x -> {asset_id}_upscaled.png
        """
        raw_asset_path = Path(raw_asset_path)
        cleaned_path = self.output_dir / f"{asset_id}_cleaned.png"
        upscaled_path = self.output_dir / f"{asset_id}_upscaled.png"

        current_path = raw_asset_path
        if remove_bg:
            current_path = self.remove_background(current_path, cleaned_path)
        
        if upscale:
            current_path = self.upscale_real_esrgan(current_path, upscaled_path, scale=4)

        return current_path
