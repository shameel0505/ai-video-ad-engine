import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
KLING_API_KEY = os.getenv("KLING_API_KEY", "")
LUMA_API_KEY = os.getenv("LUMA_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Storage Paths
STORAGE_DIR = BASE_DIR / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"
BRANDS_DIR = STORAGE_DIR / "brands"
OUTPUT_DIR = STORAGE_DIR / "outputs"

# Ensure all storage directories exist
for p in [STORAGE_DIR, UPLOAD_DIR, PROCESSED_DIR, BRANDS_DIR, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

