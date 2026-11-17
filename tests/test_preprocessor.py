"""
Verification tests for Module 3A: Preprocessor (src/generation/preprocessor.py)
"""

import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil

# Ensure workspace src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.preprocessor import AssetPreprocessor


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.preprocessor = AssetPreprocessor(output_dir=self.test_dir)
        
        # Create a mock source image
        self.input_image = Path(self.test_dir) / "test_product.jpg"
        self.input_image.write_bytes(b"DUMMY_RAW_IMAGE_DATA_1234567890")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_rembg_normal_execution(self):
        mock_rembg = MagicMock()
        mock_rembg.remove.return_value = b"REMBG_CLEANED_TRANSPARENT_PNG_BYTES"

        with patch.dict("sys.modules", {"rembg": mock_rembg}):
            out_png = Path(self.test_dir) / "output_cleaned.png"
            result = self.preprocessor.remove_background(self.input_image, out_png)
            self.assertTrue(result.exists())
            self.assertEqual(result.read_bytes(), b"REMBG_CLEANED_TRANSPARENT_PNG_BYTES")
            mock_rembg.remove.assert_called_once()

    def test_rembg_crash_fallback_to_original(self):
        # Force rembg import failure or execution exception
        mock_rembg = MagicMock()
        mock_rembg.remove.side_effect = RuntimeError("u2net onnx model execution failed")

        with patch.dict("sys.modules", {"rembg": mock_rembg}):
            out_png = Path(self.test_dir) / "output_fallback.png"
            result = self.preprocessor.remove_background(self.input_image, out_png)
            self.assertTrue(result.exists())
            self.assertEqual(result.read_bytes(), b"DUMMY_RAW_IMAGE_DATA_1234567890")

    @patch("httpx.get")
    def test_replicate_normal_execution(self, mock_http_get):
        mock_replicate = MagicMock()
        mock_replicate.run.return_value = "https://replicate.delivery/pbxt/test_output.png"
        
        mock_response = MagicMock()
        mock_response.content = b"UPSCALED_4X_IMAGE_BYTES"
        mock_response.raise_for_status = MagicMock()
        mock_http_get.return_value = mock_response

        with patch.dict("sys.modules", {"replicate": mock_replicate}):
            out_upscaled = Path(self.test_dir) / "output_upscaled.png"
            result = self.preprocessor.upscale_real_esrgan(self.input_image, out_upscaled)

            self.assertTrue(result.exists())
            self.assertEqual(result.read_bytes(), b"UPSCALED_4X_IMAGE_BYTES")
            mock_replicate.run.assert_called_once()

    @patch("time.sleep", return_value=None)
    def test_replicate_failure_fallback_after_retry(self, mock_sleep):
        mock_replicate = MagicMock()
        mock_replicate.run.side_effect = RuntimeError("Replicate API rate limit exceeded")

        with patch.dict("sys.modules", {"replicate": mock_replicate}):
            out_upscaled = Path(self.test_dir) / "output_fallback_upscale.png"
            result = self.preprocessor.upscale_real_esrgan(self.input_image, out_upscaled)

            self.assertTrue(result.exists())
            # Should fall back to the original input image bytes
            self.assertEqual(result.read_bytes(), b"DUMMY_RAW_IMAGE_DATA_1234567890")
            self.assertEqual(mock_replicate.run.call_count, 2)
            mock_sleep.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
