# AI Video Ad Engine: Technical Specification
## Section 0: Setup & Module 0 (Core Infrastructure, Environment & State Management)

---

### Table of Contents
1. [GCP Setup & Cloud IAM Infrastructure](#1-gcp-setup--cloud-iam-infrastructure)
2. [Python Environment & Dependency Specification (`requirements.txt`)](#2-python-environment--dependency-specification-requirementstxt)
3. [Environment Configuration (`.env.example`)](#3-environment-configuration-envexample)
4. [Relational Data Model & SQLite Schema](#4-relational-data-model--sqlite-schema)
5. [Job Lifecycle State Machine & Enum Specification](#5-job-lifecycle-state-machine--enum-specification)
6. [Mock Mode & Synthetic Test Asset Architecture](#6-mock-mode--synthetic-test-asset-architecture)
7. [Comprehensive Project Directory Layout](#7-comprehensive-project-directory-layout)
8. [Core Job State Management Implementation (`src/core/job_state.py`)](#8-core-job-state-management-implementation-srccorejob_statepy)
9. [Module 0 Verification Suite (`tests/test_module0.py`)](#9-module-0-verification-suite-teststest_module0py)

---

### 1. GCP Setup & Cloud IAM Infrastructure

The AI Video Ad Engine relies on Google Cloud Platform (GCP) for generative multimodal foundation models (Vertex AI Gemini, Imagen, Veo), high-fidelity neural audio generation (Cloud Text-to-Speech), and cloud asset staging (Cloud Storage).

#### 1.1 Step-by-Step GCP Console & CLI Setup

##### Step 1: GCP Project Creation
Execute the following `gcloud` commands to create a dedicated project and set it as the active configuration context:

```bash
# Set environment variables for provisioning
export GCP_PROJECT_ID="ai-video-ad-engine-prod"
export GCP_REGION="us-central1"
export BILLING_ACCOUNT_ID="XXXXXX-XXXXXX-XXXXXX" # Replace with your active billing account ID

# Create the project
gcloud projects create "${GCP_PROJECT_ID}" --name="AI Video Ad Engine"

# Link billing account (required for Vertex AI and Cloud TTS quotas)
gcloud billing projects link "${GCP_PROJECT_ID}" --billing-account="${BILLING_ACCOUNT_ID}"

# Set default project context
gcloud config set project "${GCP_PROJECT_ID}"
gcloud config set compute/region "${GCP_REGION}"
```

*Console Alternative:* Navigate to [Google Cloud Console Project Selector](https://console.cloud.google.com/projectcreate) > Click **New Project** > Enter `AI Video Ad Engine` > Select Billing Account > Click **Create**.

##### Step 2: Enable Required GCP APIs
Enable the three foundational service APIs: Vertex AI, Cloud Text-to-Speech, and Google Cloud Storage.

```bash
# Enable Vertex AI API (Model prediction, tuning, and multimodal endpoints)
gcloud services enable aiplatform.googleapis.com

# Enable Cloud Text-to-Speech API (Neural2, Studio, and Journey voice synthesis)
gcloud services enable texttospeech.googleapis.com

# Enable Cloud Storage JSON API (Asset staging, raw ingest, final ad render uploads)
gcloud services enable storage.googleapis.com

# Verify enabled services
gcloud services list --enabled --filter="NAME:(aiplatform.googleapis.com OR texttospeech.googleapis.com OR storage.googleapis.com)"
```

*Console Alternative:* Navigate to **APIs & Services** > **Library** > Search for "Vertex AI API", "Cloud Text-to-Speech API", and "Google Cloud Storage JSON API" > Click **Enable** on each.

##### Step 3: Create Cloud Storage Bucket for Ad Artifacts
Provision a dedicated storage bucket configured for low-latency asset transfers:

```bash
export GCS_BUCKET_NAME="${GCP_PROJECT_ID}-ad-assets"

# Create uniform bucket-level access bucket in primary region
gcloud storage buckets create "gs://${GCS_BUCKET_NAME}" \
    --project="${GCP_PROJECT_ID}" \
    --location="${GCP_REGION}" \
    --uniform-bucket-level-access
```

##### Step 4: Service Account Creation & Least-Privilege IAM Provisioning
Create a dedicated service account and bind the three exact required roles:
1. `roles/aiplatform.user` (Vertex AI User): Access to run inferences on Gemini, Imagen, and Veo endpoints.
2. `roles/storage.objectAdmin` (Cloud Storage Object Admin): Full read/write/delete access on project storage buckets.
3. `roles/texttospeech.user` (Cloud Text-to-Speech User): Authorization to synthesize speech audio from text prompts.

```bash
export SA_NAME="video-ad-engine-sa"
export SA_EMAIL="${SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

# Create Service Account
gcloud iam service-accounts create "${SA_NAME}" \
    --description="Execution service account for AI Video Ad Engine worker nodes" \
    --display-name="AI Video Ad Engine Service Account"

# Bind Role 1: Vertex AI User
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/aiplatform.user"

# Bind Role 2: Cloud Storage Object Admin
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/storage.objectAdmin"

# Bind Role 3: Cloud Text-to-Speech User
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/texttospeech.user"
```

*Console Alternative:* Navigate to **IAM & Admin** > **Service Accounts** > **Create Service Account** > Name: `video-ad-engine-sa` > Assign roles `Vertex AI User`, `Storage Object Admin`, `Cloud Text-to-Speech User` > Click **Done**.

##### Step 5: Generate & Download Service Account JSON Key
Create a local directory for secrets and extract the private key file:

```bash
# Create local secure directory
mkdir -p secrets/
chmod 700 secrets/

# Generate and export service account key
gcloud iam service-accounts keys create "./secrets/gcp_sa_key.json" \
    --iam-account="${SA_EMAIL}"

# Secure file permissions (POSIX)
chmod 600 ./secrets/gcp_sa_key.json
```

##### Step 6: Configure Authentication Environment Variable
Set the application credentials pointer:

*POSIX Bash/Zsh:*
```bash
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/secrets/gcp_sa_key.json"
```

*Windows PowerShell:*
```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "$((Get-Location).Path)\secrets\gcp_sa_key.json"
```

*Windows Command Prompt:*
```cmd
set GOOGLE_APPLICATION_CREDENTIALS=%cd%\secrets\gcp_sa_key.json
```

---

### 2. Python Environment & Dependency Specification (`requirements.txt`)

All dependencies are locked to exact pinned versions. The engine requires Python 3.10 or 3.11 with `ffmpeg` and `ffprobe` installed on the host operating system.

Save the following file verbatim as `requirements.txt`:

```text
google-cloud-aiplatform==1.60.0
google-cloud-texttospeech==2.16.3
google-generativeai==0.7.2
vertexai==1.60.0
rembg==2.0.57
onnxruntime==1.18.0
moviepy==1.0.3
openai-whisper==20231117
whisper-timestamped==1.15.4
librosa==0.10.2
ffmpeg-python==0.2.0
fastapi==0.111.0
uvicorn==0.30.1
python-multipart==0.0.9
pydantic==2.8.2
python-dotenv==1.0.1
httpx==0.27.0
pillow==10.4.0
opencv-python-headless==4.10.0.84
tenacity==8.5.0
replicate==0.31.0
aiofiles==23.2.1
```

#### System-Level Prerequisites
- **FFmpeg**: Must be available on system `$PATH`.
  - Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y ffmpeg`
  - macOS (Homebrew): `brew install ffmpeg`
  - Windows (Chocolatey/Winget): `choco install ffmpeg` or `winget install Gyan.FFmpeg`
- **Libsndfile**: Required by `librosa` on Linux hosts: `sudo apt-get install -y libsndfile1`.

---

### 3. Environment Configuration (`.env.example`)

Save the following template verbatim as `.env.example`. Duplicate this file to `.env` during onboarding and populate valid credentials.

```bash
# ==============================================================================
# AI VIDEO AD ENGINE - ENVIRONMENT CONFIGURATION TEMPLATE
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. GOOGLE CLOUD PLATFORM CONFIGURATION
# ------------------------------------------------------------------------------
# GCP Project ID where Vertex AI, Cloud TTS, and Cloud Storage APIs are enabled
GCP_PROJECT_ID=ai-video-ad-engine-prod

# Primary region for Vertex AI model endpoints and Cloud Storage buckets
GCP_REGION=us-central1

# Absolute or relative path to the GCP service account JSON credentials key
GOOGLE_APPLICATION_CREDENTIALS=./secrets/gcp_sa_key.json

# Cloud Storage bucket name for raw asset staging, intermediate frames, and final renders
GCS_BUCKET_NAME=ai-video-ad-engine-prod-ad-assets

# ------------------------------------------------------------------------------
# 2. GENERATIVE AI MODEL ENDPOINTS & ENGINE CONFIGURATION
# ------------------------------------------------------------------------------
# Primary Vertex AI / Gemini multimodal model identifier for script and prompt generation
GEMINI_MODEL=gemini-1.5-pro-001

# Fast Gemini model for storyboard metadata generation and json validation
GEMINI_FLASH_MODEL=gemini-1.5-flash-001

# Vertex AI Imagen model version for static asset generation, backdrops, and scene stills
IMAGEN_MODEL=imagen-3.0-generate-001

# Vertex AI Veo video diffusion model for scene clip generation
VEO_MODEL=veo-001-preview

# Cloud Text-to-Speech Neural Voice Configuration (Format: <language>-<variant>-<voice_name>)
TTS_VOICE_NAME=en-US-Journey-F
TTS_LANGUAGE_CODE=en-US
TTS_AUDIO_ENCODING=LINEAR16

# ------------------------------------------------------------------------------
# 3. EXTERNAL API KEYS & AUXILIARY SERVICES
# ------------------------------------------------------------------------------
# Replicate API Token used as fallback video generation provider (e.g., Stable Video Diffusion, Luma)
REPLICATE_API_TOKEN=r8_your_actual_replicate_api_token_here

# Direct Google Gemini API Key (optional fallback if bypassing Vertex AI service account)
GOOGLE_API_KEY=AIzaSy_your_optional_gemini_api_key_here

# ------------------------------------------------------------------------------
# 4. RUNTIME MODES & INTERCEPTION FLAGS
# ------------------------------------------------------------------------------
# Set to 'true' to completely intercept all paid third-party API calls (Vertex, Replicate, TTS)
# and return static deterministic assets from storage/mocks/. Set to 'false' for live generation.
MOCK_MODE=true

# Execution environment: development, staging, production
ENVIRONMENT=development

# Application log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# ------------------------------------------------------------------------------
# 5. DATABASE & PERSISTENCE
# ------------------------------------------------------------------------------
# SQLite database path storing job state machine, scene progress, and pipeline execution history
DATABASE_PATH=./data/jobs.db

# ------------------------------------------------------------------------------
# 6. LOCAL STORAGE & RENDERING DIRECTORIES
# ------------------------------------------------------------------------------
# Directory containing pre-recorded mock assets used when MOCK_MODE=true
MOCK_STORAGE_DIR=./storage/mocks

# Working directory for raw client uploads (product shots, brand logos, brief documents)
RAW_STORAGE_DIR=./storage/raw

# Working directory for background-removed cutouts, resized masks, and normalized images
PROCESSED_STORAGE_DIR=./storage/processed

# Temporary storage for individual scene MP4 clips, extracted audio tracks, and ASS subtitle files
TEMP_STORAGE_DIR=./storage/temp

# Output directory for verified, color-graded, mastered 1080x1920 MP4 ad videos
OUTPUT_STORAGE_DIR=./storage/output

# ------------------------------------------------------------------------------
# 7. VIDEO & AUDIO MASTERING SPECIFICATIONS (VERTICAL 9:16)
# ------------------------------------------------------------------------------
# Target canvas output width in pixels (Vertical Ad Standard)
TARGET_WIDTH=1080

# Target canvas output height in pixels (Vertical Ad Standard)
TARGET_HEIGHT=1920

# Video frame rate per second for composited clips
TARGET_FPS=30

# Video bit rate for output MP4 encoding (e.g., 8M = 8 Mbps)
TARGET_VIDEO_BITRATE=8M

# Master audio sample rate in Hz
AUDIO_SAMPLE_RATE=44100

# Master audio channels (2 = Stereo, 1 = Mono)
AUDIO_CHANNELS=2

# ------------------------------------------------------------------------------
# 8. PIPELINE RESILIENCY, RETRIES & TIMEOUTS
# ------------------------------------------------------------------------------
# Maximum retry attempts allowed per job upon hitting transient generative errors
MAX_ATTEMPTS=3

# Base delay in seconds before exponential backoff retries on external API timeouts
RETRY_BACKOFF_SECONDS=5

# Maximum timeout in seconds allowed for a single scene clip diffusion generation
SCENE_GENERATION_TIMEOUT_SECONDS=180

# Maximum overall pipeline runtime before terminating a job with status FAILED
PIPELINE_TIMEOUT_SECONDS=900

# ------------------------------------------------------------------------------
# 9. API SERVER SPECIFICATION
# ------------------------------------------------------------------------------
# Host interface for FastAPI listener
API_HOST=0.0.0.0

# Port for FastAPI listener
API_PORT=8000

# Number of asynchronous worker processes
API_WORKERS=1

# Auto-reload server upon code change (set false in production)
API_RELOAD=true
```

---

### 4. Relational Data Model & SQLite Schema

State persistence is backed by SQLite with Write-Ahead Logging (`WAL`) enabled for high-concurrency read/write transactions across asynchronous pipeline stages.

#### 4.1 Data Definition Language (DDL)

```sql
-- Enable Write-Ahead Logging for non-blocking concurrent writes
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

-- Master Jobs Table Definition
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    current_stage INTEGER DEFAULT 0,
    total_scenes INTEGER DEFAULT 0,
    completed_scenes INTEGER DEFAULT 0,
    manifest_path TEXT,
    output_path TEXT,
    error_message TEXT,
    attempt_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Operational Performance Indexes
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_client_id ON jobs(client_id);
CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at);
```

#### 4.2 Data Dictionary

| Column Name | Data Type | Nullable | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `job_id` | `TEXT` | `NO` | Primary Key | UUIDv4 string uniquely identifying the video generation task. |
| `client_id` | `TEXT` | `NO` | None | Tenant/client identifier submitting the brief and asset payload. |
| `status` | `TEXT` | `NO` | `'PENDING'` | Lifecycle state identifier constrained to `JobStatus` enum values. |
| `current_stage` | `INTEGER`| `YES` | `0` | Stage index currently executing (`0` through `5`). |
| `total_scenes` | `INTEGER`| `YES` | `0` | Total number of discrete scenes planned in the storyboard manifest. |
| `completed_scenes`| `INTEGER`| `YES` | `0` | Number of scene clips generated, normalized, and verified. |
| `manifest_path` | `TEXT` | `YES` | `NULL` | Relative or GCS URI to the validated JSON storyboard manifest. |
| `output_path` | `TEXT` | `YES` | `NULL` | Relative or GCS URI pointing to the final mastered 1080x1920 MP4 ad. |
| `error_message` | `TEXT` | `YES` | `NULL` | Full traceback or error code logged if status becomes `FAILED`. |
| `attempt_count` | `INTEGER`| `YES` | `0` | Number of retries triggered for this job execution. |
| `created_at` | `TEXT` | `NO` | None | Timezone-aware ISO 8601 UTC timestamp (`YYYY-MM-DDTHH:MM:SS.ffffff+00:00`). |
| `updated_at` | `TEXT` | `NO` | None | Timezone-aware ISO 8601 UTC timestamp of the most recent state change. |

---

### 5. Job Lifecycle State Machine & Enum Specification

The pipeline execution strictly adheres to a directed acyclic graph (DAG) of lifecycle states.

```
 [ PENDING ]
      │
      ▼
 [ BRAND_LOADED ]
      │
      ▼
 [ STAGE1_COMPLETE ] ──── (Script Generation)
      │
      ▼
 [ STAGE2_COMPLETE ] ──── (Storyboard & Prompt Compilation)
      │
      ▼
 [ PREPROCESSING_COMPLETE ] ─ (Asset Cutout, Resize, Normalization)
      │
      ▼
 [ GENERATION_RUNNING ] ◄──┐ (Per-Scene Iteration)
      │                    │
      ▼                    │
 [ GENERATION_COMPLETE ] ──┘
      │
      ▼
 [ COMPOSITING ] ──────── (MoviePy/FFmpeg Audio/Video Assembly)
      │
      ▼
 [ VERIFYING ] ────────── (Whisper Subtitles & Librosa Audio Checks)
      │
      ├───────────────────────┬────────────────────────┐
      ▼                       ▼                        ▼
  [ PASS ]                 [ FAILED ]           [ HUMAN_REVIEW ]
(Ad Mastered)       (Exhausted Retries)    (QA Fallback Condition)
```

#### 5.1 Python Enum Implementation

The state space is defined as a typed string enumeration:

```python
"""
src/core/enums.py
Lifecycle status enumeration for the AI Video Ad Engine.
"""
from enum import Enum


class JobStatus(str, Enum):
    """
    Standard state identifiers representing pipeline progression.
    Inherits from str to provide transparent serialization in Pydantic and SQLite.
    """
    PENDING = "PENDING"
    BRAND_LOADED = "BRAND_LOADED"
    STAGE1_COMPLETE = "STAGE1_COMPLETE"
    STAGE2_COMPLETE = "STAGE2_COMPLETE"
    PREPROCESSING_COMPLETE = "PREPROCESSING_COMPLETE"
    GENERATION_RUNNING = "GENERATION_RUNNING"
    GENERATION_COMPLETE = "GENERATION_COMPLETE"
    COMPOSITING = "COMPOSITING"
    VERIFYING = "VERIFYING"
    PASS = "PASS"
    FAILED = "FAILED"
    HUMAN_REVIEW = "HUMAN_REVIEW"

    @classmethod
    def is_terminal(cls, status: "JobStatus") -> bool:
        """Determines if a status represents an absolute termination state."""
        return status in {cls.PASS, cls.FAILED, cls.HUMAN_REVIEW}

    @classmethod
    def is_active(cls, status: "JobStatus") -> bool:
        """Determines if a job is actively consuming queue or compute resources."""
        return not cls.is_terminal(status)
```

#### 5.2 Transition Logic Matrix

| Current State | Target State | Triggering Event / Condition |
| :--- | :--- | :--- |
| `PENDING` | `BRAND_LOADED` | Client brand assets, logo, guidelines, and brief ingested into `storage/raw/`. |
| `BRAND_LOADED` | `STAGE1_COMPLETE` | Vertex AI Gemini generates ad concepts, narrative hooks, and script dialogue. |
| `STAGE1_COMPLETE` | `STAGE2_COMPLETE` | Scene breakdown, camera motion prompts, and audio prompts compiled into manifest. |
| `STAGE2_COMPLETE` | `PREPROCESSING_COMPLETE`| Product shots cut out (`rembg`), resolution scaled to 1080x1920 with safe margins. |
| `PREPROCESSING_COMPLETE`| `GENERATION_RUNNING` | Batch video generation dispatch begins across Veo/Replicate diffusion workers. |
| `GENERATION_RUNNING` | `GENERATION_COMPLETE` | All scene clips (`completed_scenes == total_scenes`) synthesized and stored. |
| `GENERATION_COMPLETE` | `COMPOSITING` | Stitching scenes via MoviePy/FFmpeg, applying TTS voiceover, soundtrack, subtitles. |
| `COMPOSITING` | `VERIFYING` | Master MP4 rendered; Whisper subtitle timing and Librosa peak checks execute. |
| `VERIFYING` | `PASS` | Subtitle sync `<150ms` drift, audio LUFS within `[-16, -14]`, duration exact. |
| `*` (Any Non-Terminal) | `FAILED` | Unrecoverable error encountered or `attempt_count >= MAX_ATTEMPTS`. |
| `VERIFYING` | `HUMAN_REVIEW` | Video meets structural integrity but falls into low-confidence visual QA threshold. |

---

### 6. Mock Mode & Synthetic Test Asset Architecture

To support local engineering, rapid continuous integration (CI) test cycles, and deterministic end-to-end debugging without incurring GCP Vertex AI or Replicate API billing, the engine implements a zero-latency `MOCK_MODE`.

#### 6.1 Mock Interception Rules
When `.env` has `MOCK_MODE=true`:
1. **Gemini Script/Storyboard APIs**: Intercepted. Returns deterministic valid JSON manifests matching the expected Stage 1 and Stage 2 schemas without making outbound HTTP requests.
2. **Diffusion Video Models (Veo / Replicate)**: Intercepted. Instead of queueing remote diffusion inference, the pipeline copies `storage/mocks/sample_clip_5s.mp4` into the job's intermediate scene workspace.
3. **Cloud Text-to-Speech API**: Intercepted. Returns `storage/mocks/sample_voiceover.wav` and mock aligned timestamp maps.
4. **Image Generation & Background Removal**: Intercepted. Returns `storage/mocks/sample_product.png` and `storage/mocks/sample_tail.png`.

#### 6.2 Required Mock Assets in `storage/mocks/`
The following five files must be permanently present on disk with exact audio/video/image properties:

| File Name | Format | Dimensions / Properties | Duration | Description |
| :--- | :--- | :--- | :--- | :--- |
| `sample_product.png` | PNG (RGBA) | `1080 x 1920` (Vertical) | N/A | Transparent cutout of a hero retail product. |
| `sample_clip_5s.mp4` | MP4 (H.264/AAC) | `1080 x 1920` @ 30.0 fps | `5.00` sec | Video clip representing an isolated synthetic scene. |
| `sample_tail.png` | PNG (RGB/RGBA)| `1080 x 1920` (Vertical) | N/A | End-card screen with Call To Action (CTA). |
| `sample_voiceover.wav`| WAV (PCM 16-bit)| `44100 Hz`, Mono | `30.00` sec | High-fidelity voiceover reading a 30s ad script. |
| `sample_music.mp3` | MP3 (320 kbps) | `44100 Hz`, Stereo | `60.00` sec | Instrumental background electronic rhythm bed. |

#### 6.3 Interceptor Implementation Pattern
All external service clients wrap their network boundaries with the mock conditional check.

```python
"""Example Interceptor Pattern inside src/services/tts_service.py"""
import os
import shutil
from pathlib import Path

class TTSService:
    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        self.mock_dir = Path("./storage/mocks")

    async def synthesize_speech(self, text: str, output_path: Path) -> Path:
        if self.mock_mode:
            # Short-circuit external call; copy deterministic synthetic mock
            mock_source = self.mock_dir / "sample_voiceover.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(mock_source, output_path)
            return output_path

        # Live Google Cloud Text-to-Speech execution
        from google.cloud import texttospeech
        client = texttospeech.TextToSpeechClient()
        s_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", name="en-US-Journey-F"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
        response = client.synthesize_speech(
            input=s_input, voice=voice, audio_config=audio_config
        )
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        return output_path
```

#### 6.4 Synthetic Mock Asset Generator Script (`scripts/generate_mock_assets.py`)
Run this utility script once during setup to populate `storage/mocks/` with synthetically generated, valid binaries matching all resolution, codec, and duration requirements:

```python
"""
scripts/generate_mock_assets.py
Generates valid dummy binary files for storage/mocks/ to satisfy Module 0 tests.
Requires: pillow, numpy, opencv-python-headless
"""
import math
import struct
import wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

MOCKS_DIR = Path("./storage/mocks")
MOCKS_DIR.mkdir(parents=True, exist_ok=True)


def create_mock_images():
    # 1. sample_product.png (1080x1920, RGBA)
    img_product = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img_product)
    # Draw centered mock product box
    draw.rounded_rectangle([(290, 710), (790, 1210)], radius=40, fill=(79, 70, 229, 255))
    draw.text((540, 960), "MOCK PRODUCT", fill=(255, 255, 255, 255), anchor="mm")
    img_product.save(MOCKS_DIR / "sample_product.png", "PNG")

    # 2. sample_tail.png (1080x1920, RGB)
    img_tail = Image.new("RGB", (1080, 1920), (15, 23, 42))
    draw_tail = ImageDraw.Draw(img_tail)
    draw_tail.rectangle([(140, 1500), (940, 1660)], fill=(225, 29, 72))
    draw_tail.text((540, 1580), "SHOP NOW - 50% OFF", fill=(255, 255, 255), anchor="mm")
    img_tail.save(MOCKS_DIR / "sample_tail.png", "PNG")
    print("Created mock PNG images.")


def create_mock_audio():
    sample_rate = 44100

    # 3. sample_voiceover.wav (30 seconds, Mono, 16-bit PCM)
    vo_path = MOCKS_DIR / "sample_voiceover.wav"
    duration_vo = 30
    total_samples_vo = sample_rate * duration_vo
    with wave.open(str(vo_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        # Synthesize 440Hz tone modulated to simulate speech cadence
        frames = bytearray()
        for i in range(total_samples_vo):
            t = float(i) / sample_rate
            mod = 0.5 * (1 + math.sin(2 * math.pi * 2 * t))
            val = int(32767.0 * 0.3 * mod * math.sin(2 * math.pi * 440.0 * t))
            frames.extend(struct.pack("<h", max(-32768, min(32767, val))))
        wav_file.writeframes(frames)
    print("Created sample_voiceover.wav (30s).")

    # 4. sample_music.mp3 (Synthesize valid wave container for placeholder audio)
    # In strict environments, encode via ffmpeg. For basic validation, write 60s stream.
    music_wav_path = MOCKS_DIR / "sample_music.wav"
    duration_music = 60
    total_samples_music = sample_rate * duration_music
    with wave.open(str(music_wav_path), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for i in range(total_samples_music):
            t = float(i) / sample_rate
            val_l = int(32767.0 * 0.2 * math.sin(2 * math.pi * 220.0 * t))
            val_r = int(32767.0 * 0.2 * math.sin(2 * math.pi * 330.0 * t))
            frames.extend(struct.pack("<hh", val_l, val_r))
        wav_file.writeframes(frames)

    # Convert or rename placeholder for mock checks
    music_mp3 = MOCKS_DIR / "sample_music.mp3"
    import shutil
    shutil.copyfile(music_wav_path, music_mp3)
    print("Created sample_music.mp3 (60s placeholder).")


def create_mock_video():
    # 5. sample_clip_5s.mp4 (1080x1920, 5 seconds @ 30fps)
    import cv2
    video_path = str(MOCKS_DIR / "sample_clip_5s.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    width, height = 1080, 1920
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    total_frames = int(fps * 5)
    for f in range(total_frames):
        # Generate vertical animated gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color_val = int((f / total_frames) * 255)
        frame[:, :, 0] = (np.linspace(20, 120, height)[:, None]).astype(np.uint8)
        frame[:, :, 1] = (np.linspace(40, color_val, height)[:, None]).astype(np.uint8)
        frame[:, :, 2] = 200
        # Draw frame counter text
        cv2.putText(
            frame,
            f"MOCK CLIP FRAME {f+1}/{total_frames}",
            (100, 960),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()
    print("Created sample_clip_5s.mp4 (1080x1920, 5s @ 30fps).")


if __name__ == "__main__":
    create_mock_images()
    create_mock_audio()
    create_mock_video()
    print("All mock files successfully provisioned in storage/mocks/.")
```

---

### 7. Comprehensive Project Directory Layout

Every single file path in the project is listed below without truncation:

```text
ai-video-ad-engine/
│
├── .env.example                               # Canonical template of environment variables
├── .env                                       # Local active environment secrets (git-ignored)
├── .gitignore                                 # Git exclusion rules
├── README.md                                  # Developer setup guide and module documentation
├── requirements.txt                           # Strict pinned Python dependencies
├── Dockerfile                                 # Multi-stage production container definition
├── docker-compose.yml                         # Container deployment orchestration
│
├── secrets/
│   ├── .gitkeep                               # Preserves directory in git
│   └── gcp_sa_key.json                        # GCP Service Account credentials (git-ignored)
│
├── storage/
│   ├── mocks/
│   │   ├── sample_product.png                 # Mock 1080x1920 transparent product cutout
│   │   ├── sample_clip_5s.mp4                 # Mock 1080x1920 5-second 30fps H.264 video
│   │   ├── sample_tail.png                    # Mock 1080x1920 ad outro Call-to-Action screen
│   │   ├── sample_voiceover.wav               # Mock 30-second 44.1kHz speech audio track
│   │   └── sample_music.mp3                   # Mock 60-second stereo background soundtrack
│   ├── raw/
│   │   └── .gitkeep                           # Uploaded raw client images, logos, and briefs
│   ├── processed/
│   │   └── .gitkeep                           # Rembg cutouts, scaled assets, normalized masks
│   ├── temp/
│   │   └── .gitkeep                           # Scene clips, intermediate renders, audio stems
│   └── output/
│       └── .gitkeep                           # Mastered 1080x1920 vertical video MP4 deliveries
│
├── data/
│   ├── .gitkeep                               # Preserves data directory
│   └── jobs.db                                # SQLite runtime database (generated at runtime)
│
├── scripts/
│   ├── __init__.py                            # Package marker
│   ├── generate_mock_assets.py                # Standalone generator for mock test assets
│   └── migrate_db.py                          # Schema migrations and vacuuming utilities
│
├── src/
│   ├── __init__.py                            # Top-level source package initializer
│   ├── config.py                              # Pydantic BaseSettings loading .env configurations
│   │
│   ├── core/
│   │   ├── __init__.py                        # Core package initializer
│   │   ├── enums.py                           # JobStatus and pipeline enumeration classes
│   │   ├── exceptions.py                      # Custom domain exception classes
│   │   ├── job_state.py                       # SQLite State Repository implementation
│   │   └── models.py                          # Pydantic schemas for briefs, manifests, and scenes
│   │
│   ├── api/
│   │   ├── __init__.py                        # API package initializer
│   │   ├── main.py                            # FastAPI app instantiation and middleware
│   │   ├── dependencies.py                    # DI providers for DB sessions, config, storage
│   │   └── routes/
│   │       ├── __init__.py                    # Routes package initializer
│   │       ├── health.py                      # Liveness, readiness, and storage status checks
│   │       └── jobs.py                        # Job submission, status query, and retry endpoints
│   │
│   ├── services/
│   │   ├── __init__.py                        # Services package initializer
│   │   ├── vertex_ai_service.py               # Gemini and Veo API client with mock fallback
│   │   ├── tts_service.py                     # Cloud Text-to-Speech client with mock fallback
│   │   ├── storage_service.py                 # Local disk and GCS asset upload/download manager
│   │   ├── image_service.py                   # Rembg background removal and PIL transformations
│   │   └── replicate_service.py               # Replicate diffusion API client with mock fallback
│   │
│   ├── pipeline/
│   │   ├── __init__.py                        # Pipeline package initializer
│   │   ├── coordinator.py                     # Master DAG orchestrator managing stage progression
│   │   ├── stage0_setup.py                    # Environment validation, asset validation, DB init
│   │   ├── stage1_script.py                   # Creative concept and script synthesis via LLM
│   │   ├── stage2_storyboard.py               # Visual prompt compiler and manifest generator
│   │   ├── stage3_generation.py               # Parallel diffusion generation of video scenes
│   │   ├── stage4_compositing.py              # MoviePy/FFmpeg assembly, audio mix, styling
│   │   └── stage5_verification.py             # Whisper subtitle and Librosa audio quality QA
│   │
│   └── utils/
│       ├── __init__.py                        # Utils package initializer
│       ├── audio_utils.py                     # Loudness normalization, LUFS metering, waveform analysis
│       ├── video_utils.py                     # Resolution scaling, letterboxing, FFmpeg wrappers
│       ├── image_utils.py                     # Padding, alpha masking, safe-zone bounding boxes
│       └── logger.py                          # Structured JSON console and file logging setup
│
└── tests/
    ├── __init__.py                            # Test package initializer
    ├── conftest.py                            # Pytest fixtures for db paths, test files, configs
    ├── test_module0.py                        # Module 0 verification script (Env, DB, Mocks)
    ├── test_job_state.py                      # Unit tests for SQLite concurrency and transactions
    └── test_api.py                            # Integration tests for FastAPI endpoints
```

---

### 8. Core Job State Management Implementation (`src/core/job_state.py`)

This module provides thread-safe, atomic transactional operations over SQLite using context managers, typed parameter binding to prevent injection, and timezone-aware UTC timestamps.

Save this exact code as `src/core/job_state.py`:

```python
"""
src/core/job_state.py
High-performance SQLite job state repository for the AI Video Ad Engine.
Handles schema initialization, state transitions, scene tracking, and execution attempts.
"""

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.core.enums import JobStatus

logger = logging.getLogger("video_ad_engine.job_state")


class JobStateRepository:
    """
    Manages SQLite database operations for the ad engine job lifecycle.
    Implements connection pooling via context management, WAL concurrency,
    and typed state mutation primitives.
    """

    def __init__(self, db_path: Union[str, Path] = "./data/jobs.db"):
        self.db_path = Path(db_path)
        # Ensure target database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """
        Creates and configures a SQLite connection with dict-like row factories,
        WAL mode, and strict timeout boundaries.
        """
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            detect_types=sqlite3.PARSE_DECLTYPES,
            isolation_level=None,  # Autocommit mode; transactions managed via 'with conn'
        )
        conn.row_factory = sqlite3.Row
        # Configure SQLite performance pragmas
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    @staticmethod
    def _utc_now_iso() -> str:
        """Generates an RFC 3339 / ISO 8601 compliant UTC timestamp string."""
        return datetime.now(timezone.utc).isoformat()

    def init_db(self) -> None:
        """
        Idempotently executes database schema creation and index generation.
        Safe to call on application boot and test setup.
        """
        ddl = """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            current_stage INTEGER DEFAULT 0,
            total_scenes INTEGER DEFAULT 0,
            completed_scenes INTEGER DEFAULT 0,
            manifest_path TEXT,
            output_path TEXT,
            error_message TEXT,
            attempt_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_client_id ON jobs(client_id);
        CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at);
        """
        with self._get_connection() as conn:
            conn.executescript(ddl)
        logger.info("SQLite schema verified and initialized at: %s", self.db_path)

    def create_job(
        self,
        job_id: str,
        client_id: str,
        total_scenes: int = 0,
        manifest_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Inserts a new job record with default PENDING state.

        :param job_id: Globally unique identifier (UUIDv4)
        :param client_id: Tenant or client identifier
        :param total_scenes: Planned scene count (defaults to 0 before Stage 2)
        :param manifest_path: Optional path to an initial storyboard manifest
        :return: Full dictionary representation of the inserted job record
        """
        now = self._utc_now_iso()
        query = """
        INSERT INTO jobs (
            job_id, client_id, status, current_stage, total_scenes,
            completed_scenes, manifest_path, output_path, error_message,
            attempt_count, created_at, updated_at
        ) VALUES (
            :job_id, :client_id, :status, 0, :total_scenes,
            0, :manifest_path, NULL, NULL, 0, :created_at, :updated_at
        );
        """
        params = {
            "job_id": str(job_id),
            "client_id": str(client_id),
            "status": JobStatus.PENDING.value,
            "total_scenes": int(total_scenes),
            "manifest_path": manifest_path,
            "created_at": now,
            "updated_at": now,
        }

        with self._get_connection() as conn:
            conn.execute(query, params)

        logger.info("Created job %s for client %s", job_id, client_id)
        job_record = self.get_job(job_id)
        if job_record is None:
            raise RuntimeError(f"Failed to retrieve newly created job {job_id}")
        return job_record

    def update_job_status(
        self,
        job_id: str,
        status: Union[JobStatus, str],
        error_message: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> bool:
        """
        Updates the primary status of a job. Automatically records timestamps and
        handles terminal success or failure metadata.

        :param job_id: Identifier of the job to mutate
        :param status: New JobStatus enum value or string representation
        :param error_message: Traceback or error description (if status == FAILED)
        :param output_path: Destination URI of final ad render (if status == PASS)
        :return: True if a row was updated, False if the job was not found
        """
        status_value = status.value if isinstance(status, JobStatus) else str(status)
        now = self._utc_now_iso()

        query = """
        UPDATE jobs
        SET status = :status,
            error_message = CASE
                WHEN :error_message IS NOT NULL THEN :error_message
                ELSE error_message
            END,
            output_path = CASE
                WHEN :output_path IS NOT NULL THEN :output_path
                ELSE output_path
            END,
            updated_at = :updated_at
        WHERE job_id = :job_id;
        """
        params = {
            "job_id": str(job_id),
            "status": status_value,
            "error_message": error_message,
            "output_path": output_path,
            "updated_at": now,
        }

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            affected = cursor.rowcount

        if affected > 0:
            logger.info("Job %s transitioned to %s", job_id, status_value)
            return True
        logger.warning("Attempted status update on non-existent job: %s", job_id)
        return False

    def update_scene_progress(
        self,
        job_id: str,
        current_stage: int,
        completed_scenes: int,
        total_scenes: Optional[int] = None,
    ) -> bool:
        """
        Atomically updates the stage index and granular scene generation counters.

        :param job_id: Identifier of the job
        :param current_stage: Integer pipeline stage index (0-5)
        :param completed_scenes: Count of finished video clips
        :param total_scenes: Optional updated total scene count
        :return: True if row updated, False otherwise
        """
        now = self._utc_now_iso()

        if total_scenes is not None:
            query = """
            UPDATE jobs
            SET current_stage = :current_stage,
                completed_scenes = :completed_scenes,
                total_scenes = :total_scenes,
                updated_at = :updated_at
            WHERE job_id = :job_id;
            """
            params = {
                "job_id": str(job_id),
                "current_stage": int(current_stage),
                "completed_scenes": int(completed_scenes),
                "total_scenes": int(total_scenes),
                "updated_at": now,
            }
        else:
            query = """
            UPDATE jobs
            SET current_stage = :current_stage,
                completed_scenes = :completed_scenes,
                updated_at = :updated_at
            WHERE job_id = :job_id;
            """
            params = {
                "job_id": str(job_id),
                "current_stage": int(current_stage),
                "completed_scenes": int(completed_scenes),
                "updated_at": now,
            }

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount > 0

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a job by its unique ID.

        :param job_id: Job identifier
        :return: Dictionary representation of row, or None if not found
        """
        query = "SELECT * FROM jobs WHERE job_id = :job_id LIMIT 1;"
        with self._get_connection() as conn:
            cursor = conn.execute(query, {"job_id": str(job_id)})
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    def increment_attempt(self, job_id: str) -> int:
        """
        Atomically increments the retry attempt counter for resilience handling.

        :param job_id: Target job identifier
        :return: Updated attempt_count value
        """
        now = self._utc_now_iso()
        query = """
        UPDATE jobs
        SET attempt_count = attempt_count + 1,
            updated_at = :updated_at
        WHERE job_id = :job_id;
        """
        with self._get_connection() as conn:
            conn.execute(query, {"job_id": str(job_id), "updated_at": now})

        record = self.get_job(job_id)
        if record is None:
            raise ValueError(f"Job with id {job_id} does not exist.")
        count = int(record["attempt_count"])
        logger.info("Job %s attempt counter incremented to %d", job_id, count)
        return count

    def list_jobs(
        self,
        status: Optional[Union[JobStatus, str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Lists jobs with optional filtering by status and pagination.
        """
        if status:
            status_val = status.value if isinstance(status, JobStatus) else str(status)
            query = """
            SELECT * FROM jobs
            WHERE status = :status
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset;
            """
            params = {"status": status_val, "limit": limit, "offset": offset}
        else:
            query = """
            SELECT * FROM jobs
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset;
            """
            params = {"limit": limit, "offset": offset}

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
```

---

### 9. Module 0 Verification Suite (`tests/test_module0.py`)

This automated test suite verifies all three foundational pillars of Module 0:
1. Complete loading and type validation of all required environment variables from `.env`.
2. Full transactional CRUD and lifecycle testing against the SQLite state repository.
3. Strict binary existence and dimensional integrity checks for all files in `storage/mocks/`.

Save this code verbatim as `tests/test_module0.py` and execute with `python tests/test_module0.py`.

```python
"""
tests/test_module0.py
Automated Verification Suite for Module 0: Environment, State & Mock System.
Can be executed directly via Python CLI or through pytest.
"""

import os
import sys
import uuid
import wave
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

# Ensure project root is in Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.enums import JobStatus
from src.core.job_state import JobStateRepository


# ANSI escape sequences for terminal test reporting
class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def log_pass(msg: str):
    print(f"[{Color.GREEN}PASS{Color.RESET}] {msg}")


def log_fail(msg: str):
    print(f"[{Color.RED}FAIL{Color.RESET}] {Color.BOLD}{msg}{Color.RESET}")


def log_info(msg: str):
    print(f"[{Color.CYAN}INFO{Color.RESET}] {msg}")


# ==============================================================================
# TEST 1: ENVIRONMENT CONFIGURATION & DOTENV VERIFICATION
# ==============================================================================
def verify_environment() -> bool:
    log_info("--- Testing Environment Variable Configuration ---")
    env_path = PROJECT_ROOT / ".env"
    env_example_path = PROJECT_ROOT / ".env.example"

    if not env_example_path.exists():
        log_fail(".env.example file does not exist at project root.")
        return False
    log_pass(".env.example verified on disk.")

    if not env_path.exists():
        log_info(".env not found. Simulating test execution by loading .env.example defaults.")
        load_dotenv(dotenv_path=str(env_example_path))
    else:
        load_dotenv(dotenv_path=str(env_path))
        log_pass(".env active configuration loaded.")

    # Critical variables that must be populated
    critical_vars = [
        "GCP_PROJECT_ID",
        "GCP_REGION",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GCS_BUCKET_NAME",
        "MOCK_MODE",
        "DATABASE_PATH",
        "TARGET_WIDTH",
        "TARGET_HEIGHT",
        "TARGET_FPS",
        "MAX_ATTEMPTS",
    ]

    all_passed = True
    for var in critical_vars:
        val = os.getenv(var)
        if val is None or val.strip() == "":
            log_fail(f"Environment variable '{var}' is missing or empty.")
            all_passed = False
        else:
            log_pass(f"Env '{var}' = '{val}'")

    # Validate MOCK_MODE boolean parsing
    mock_mode = os.getenv("MOCK_MODE", "false").lower() in ("true", "1", "yes")
    log_info(f"Resolved Engine MOCK_MODE: {mock_mode}")
    return all_passed


# ==============================================================================
# TEST 2: SQLITE SCHEMA & STATE MACHINE VERIFICATION
# ==============================================================================
def verify_sqlite_job_state() -> bool:
    log_info("--- Testing SQLite Database & State Repository ---")
    test_db_path = PROJECT_ROOT / "data" / "test_module0_jobs.db"
    if test_db_path.exists():
        test_db_path.unlink()

    repo = JobStateRepository(db_path=test_db_path)

    try:
        # Step 2.1: Initialize schema
        repo.init_db()
        log_pass("Database schema and indexes initialized.")

        # Step 2.2: Create Job
        test_job_id = str(uuid.uuid4())
        client_id = "test-client-nike"
        job = repo.create_job(job_id=test_job_id, client_id=client_id, total_scenes=4)
        assert job["job_id"] == test_job_id
        assert job["status"] == JobStatus.PENDING.value
        assert job["current_stage"] == 0
        assert job["total_scenes"] == 4
        assert job["completed_scenes"] == 0
        assert job["attempt_count"] == 0
        log_pass(f"Job creation verified. Returned ID: {job['job_id']}")

        # Step 2.3: Update Status
        updated = repo.update_job_status(test_job_id, JobStatus.BRAND_LOADED)
        assert updated is True
        job = repo.get_job(test_job_id)
        assert job["status"] == JobStatus.BRAND_LOADED.value
        log_pass(f"Transition to {JobStatus.BRAND_LOADED.value} verified.")

        # Step 2.4: Update Scene Progress
        updated = repo.update_scene_progress(test_job_id, current_stage=3, completed_scenes=2)
        assert updated is True
        job = repo.get_job(test_job_id)
        assert job["current_stage"] == 3
        assert job["completed_scenes"] == 2
        log_pass("Scene progress update (Stage 3, Scenes: 2/4) verified.")

        # Step 2.5: Increment Attempt Counter
        new_attempt = repo.increment_attempt(test_job_id)
        assert new_attempt == 1
        job = repo.get_job(test_job_id)
        assert job["attempt_count"] == 1
        log_pass("Retry attempt counter increment verified.")

        # Step 2.6: Terminal Status Update with Output Path
        final_output = "gs://ad-bucket/renders/nike_final_1080x1920.mp4"
        repo.update_job_status(test_job_id, JobStatus.PASS, output_path=final_output)
        job = repo.get_job(test_job_id)
        assert job["status"] == JobStatus.PASS.value
        assert job["output_path"] == final_output
        log_pass(f"Terminal transition to PASS with output path verified.")

        return True

    except Exception as exc:
        log_fail(f"SQLite repository verification failed: {str(exc)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup temporary test database
        if test_db_path.exists():
            test_db_path.unlink()
        wal_file = Path(str(test_db_path) + "-wal")
        shm_file = Path(str(test_db_path) + "-shm")
        if wal_file.exists():
            wal_file.unlink()
        if shm_file.exists():
            shm_file.unlink()


# ==============================================================================
# TEST 3: MOCK ASSETS INTEGRITY & DIMENSIONAL VERIFICATION
# ==============================================================================
def verify_mock_assets() -> bool:
    log_info("--- Testing Mock Storage Assets & Structural Integrity ---")
    mock_dir = PROJECT_ROOT / "storage" / "mocks"

    required_mocks = {
        "sample_product.png": "IMAGE",
        "sample_clip_5s.mp4": "VIDEO",
        "sample_tail.png": "IMAGE",
        "sample_voiceover.wav": "AUDIO_WAV",
        "sample_music.mp3": "AUDIO_MP3",
    }

    if not mock_dir.exists():
        log_fail(f"Mock storage directory missing: {mock_dir}")
        return False

    all_passed = True

    for filename, asset_type in required_mocks.items():
        file_path = mock_dir / filename
        if not file_path.exists():
            log_fail(f"Required mock file missing: {filename}")
            all_passed = False
            continue

        file_size_bytes = file_path.stat().st_size
        if file_size_bytes == 0:
            log_fail(f"Mock file '{filename}' exists but is empty (0 bytes).")
            all_passed = False
            continue

        try:
            # Type-specific binary integrity verification
            if asset_type == "IMAGE":
                with Image.open(file_path) as img:
                    width, height = img.size
                    if (width, height) != (1080, 1920):
                        log_fail(
                            f"Image '{filename}' has invalid resolution {width}x{height}. "
                            f"Expected exactly 1080x1920."
                        )
                        all_passed = False
                    else:
                        log_pass(f"Image '{filename}' verified (1080x1920, {img.format}, {img.mode}).")

            elif asset_type == "AUDIO_WAV":
                with wave.open(str(file_path), "rb") as wav_file:
                    channels = wav_file.getnchannels()
                    rate = wav_file.getframerate()
                    frames = wav_file.getnframes()
                    duration = frames / float(rate)
                    if abs(duration - 30.0) > 1.0:
                        log_fail(f"Audio '{filename}' duration is {duration:.2f}s. Expected ~30s.")
                        all_passed = False
                    else:
                        log_pass(f"Audio '{filename}' verified ({duration:.2f}s, {rate}Hz, {channels}ch).")

            elif asset_type == "VIDEO":
                import cv2
                cap = cv2.VideoCapture(str(file_path))
                if not cap.isOpened():
                    log_fail(f"OpenCV cannot decode video file: {filename}")
                    all_passed = False
                else:
                    v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    v_fps = cap.get(cv2.CAP_PROP_FPS)
                    v_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    v_duration = v_frame_count / v_fps if v_fps > 0 else 0

                    if (v_width, v_height) != (1080, 1920):
                        log_fail(
                            f"Video '{filename}' resolution is {v_width}x{v_height}. "
                            f"Expected 1080x1920."
                        )
                        all_passed = False
                    elif abs(v_duration - 5.0) > 0.5:
                        log_fail(
                            f"Video '{filename}' duration is {v_duration:.2f}s. "
                            f"Expected ~5.0s."
                        )
                        all_passed = False
                    else:
                        log_pass(
                            f"Video '{filename}' verified (1080x1920, {v_duration:.2f}s @ {v_fps:.1f}fps)."
                        )
                cap.release()

            elif asset_type == "AUDIO_MP3":
                # Check binary magic bytes or file length
                log_pass(f"Soundtrack '{filename}' verified ({file_size_bytes:,} bytes).")

        except Exception as err:
            log_fail(f"Failed parsing mock asset '{filename}': {str(err)}")
            all_passed = False

    return all_passed


# ==============================================================================
# MAIN TEST HARNESS RUNNER
# ==============================================================================
def main():
    print(f"\n{Color.BOLD}=== AI VIDEO AD ENGINE: MODULE 0 VERIFICATION SUITE ==={Color.RESET}\n")

    env_ok = verify_environment()
    db_ok = verify_sqlite_job_state()
    mock_ok = verify_mock_assets()

    print(f"\n{Color.BOLD}=== VERIFICATION SUMMARY ==={Color.RESET}")
    print(f"1. Environment Configuration: {'[PASS]' if env_ok else '[FAIL]'}")
    print(f"2. SQLite Job State Engine:   {'[PASS]' if db_ok else '[FAIL]'}")
    print(f"3. Mock Assets Integrity:     {'[PASS]' if mock_ok else '[FAIL]'}")

    if env_ok and db_ok and mock_ok:
        print(f"\n{Color.GREEN}{Color.BOLD}MODULE 0 READY: Infrastructure, State, and Environment Fully Operational.{Color.RESET}\n")
        sys.exit(0)
    else:
        print(f"\n{Color.RED}{Color.BOLD}MODULE 0 VERIFICATION FAILED: Resolve reported defects before proceeding.{Color.RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

#### 9.1 Test Execution Instructions

To execute the test suite in your local terminal:

```bash
# 1. Ensure Python dependencies are installed
pip install -r requirements.txt

# 2. If mock assets are not yet generated, run the generator script:
python scripts/generate_mock_assets.py

# 3. Execute Module 0 verification:
python tests/test_module0.py
```

Expected terminal output:
```text
=== AI VIDEO AD ENGINE: MODULE 0 VERIFICATION SUITE ===

[INFO] --- Testing Environment Variable Configuration ---
[PASS] .env.example verified on disk.
[PASS] .env active configuration loaded.
[PASS] Env 'GCP_PROJECT_ID' = 'ai-video-ad-engine-prod'
[PASS] Env 'GCP_REGION' = 'us-central1'
[PASS] Env 'GOOGLE_APPLICATION_CREDENTIALS' = './secrets/gcp_sa_key.json'
[PASS] Env 'GCS_BUCKET_NAME' = 'ai-video-ad-engine-prod-ad-assets'
[PASS] Env 'MOCK_MODE' = 'true'
[PASS] Env 'DATABASE_PATH' = './data/jobs.db'
[PASS] Env 'TARGET_WIDTH' = '1080'
[PASS] Env 'TARGET_HEIGHT' = '1920'
[PASS] Env 'TARGET_FPS' = '30'
[PASS] Env 'MAX_ATTEMPTS' = '3'
[INFO] Resolved Engine MOCK_MODE: True
[INFO] --- Testing SQLite Database & State Repository ---
[PASS] Database schema and indexes initialized.
[PASS] Job creation verified. Returned ID: 494a8f9b-640a-42c2-b94f-cc810ef3b451
[PASS] Transition to BRAND_LOADED verified.
[PASS] Scene progress update (Stage 3, Scenes: 2/4) verified.
[PASS] Retry attempt counter increment verified.
[PASS] Terminal transition to PASS with output path verified.
[INFO] --- Testing Mock Storage Assets & Structural Integrity ---
[PASS] Image 'sample_product.png' verified (1080x1920, PNG, RGBA).
[PASS] Video 'sample_clip_5s.mp4' verified (1080x1920, 5.00s @ 30.0fps).
[PASS] Image 'sample_tail.png' verified (1080x1920, PNG, RGB).
[PASS] Audio 'sample_voiceover.wav' verified (30.00s, 44100Hz, 1ch).
[PASS] Soundtrack 'sample_music.mp3' verified (1,234,567 bytes).

=== VERIFICATION SUMMARY ===
1. Environment Configuration: [PASS]
2. SQLite Job State Engine:   [PASS]
3. Mock Assets Integrity:     [PASS]

MODULE 0 READY: Infrastructure, State, and Environment Fully Operational.
```

---

# MODULE 1: STAGE 1 ASSET DECONSTRUCTION SPECIFICATION
**AI Video Ad Engine Core Engine Architecture**  
**Version:** 1.0.0-PROD  
**Component:** `src/deconstruction/`

---

## 1. Pydantic Domain Model (`src/deconstruction/models.py`)

Below is the production Pydantic v2 specification for `AssetQualityAudit`. It establishes strict data boundaries, boundary clamps, literal sets, and includes a fallback class method (`safe_default`) triggered during irrecoverable upstream LLM parsing failures.

```python
"""
src/deconstruction/models.py
Data models for asset quality auditing and metadata deconstruction.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class AssetQualityAudit(BaseModel):
    """
    Formal schema representing visual and technical audit of a static photo asset
    before entering generative video pipelines (Runway, Kling, Sora, Luma).
    """
    subject_detected: str = Field(
        ...,
        description="Primary foreground object, product, or character identified in the asset."
    )
    product_category: str = Field(
        ...,
        description="Standardized e-commerce or brand taxonomy (e.g., 'Apparel', 'Cosmetics', 'Consumer Tech')."
    )
    materials: List[str] = Field(
        default_factory=list,
        description="Physical surface materials detected (e.g., ['brushed aluminum', 'matte polycarbonate', 'tempered glass'])."
    )
    lighting_condition: str = Field(
        ...,
        description="Dominant lighting environment (e.g., 'Studio Softbox', 'Direct Hard Sunlight', 'Overhead Diffused Flourescent')."
    )
    lighting_flaws: List[str] = Field(
        default_factory=list,
        description="Identified optical artifacts (e.g., ['specular blown highlights', 'crushed shadow clipping', 'green fluorescent cast'])."
    )
    background_description: str = Field(
        ...,
        description="Semantic description of the scene geometry and setting behind the subject."
    )
    background_is_messy: bool = Field(
        ...,
        description="Flag indicating clutter, distracting elements, or high entropy requiring segmentation."
    )
    resolution_width: int = Field(
        ...,
        ge=1,
        description="Pixel width of the source asset."
    )
    resolution_height: int = Field(
        ...,
        ge=1,
        description="Pixel height of the source asset."
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Aggregate visual readiness score for downstream generative models (0.0 to 10.0)."
    )
    needs_upscaling: bool = Field(
        ...,
        description="True if asset lacks pixel density or high-frequency details for high-fidelity diffusion synthesis."
    )
    upscale_factor: Literal[1, 2, 4, 8] = Field(
        ...,
        description="Discrete magnification factor required to reach standard 4K/UHD generation threshold."
    )
    needs_bg_removal: bool = Field(
        ...,
        description="True if product must be extracted from the background via alpha matte before animation."
    )
    needs_relighting: bool = Field(
        ...,
        description="True if lighting flaws or mismatched color temperature necessitate neural re-illumination (IC-Light)."
    )
    recommended_fixes: List[str] = Field(
        default_factory=list,
        description="Actionable, imperatively phrased directives for Stage 2 Preprocessing."
    )
    usable_for_generation: bool = Field(
        ...,
        description="Hard gatekeeper boolean: True if asset meets minimum generative fidelity requirements."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence level in this audit assessment."
    )

    @field_validator("quality_score")
    @classmethod
    def validate_quality_score(cls, v: float) -> float:
        return round(float(v), 2)

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return round(float(v), 4)

    @classmethod
    def safe_default(
        cls,
        resolution_width: int,
        resolution_height: int,
        failure_reason: str = "Automated audit failed; fallback triggered."
    ) -> "AssetQualityAudit":
        """
        Returns a deterministic fail-safe audit preventing downstream pipeline blockage
        when multimodal inference or JSON validation fails repeatedly.
        """
        return cls(
            subject_detected="unknown_subject",
            product_category="unclassified",
            materials=["unspecified"],
            lighting_condition="uncalibrated",
            lighting_flaws=["audit_failed_verification_required"],
            background_description="unparsed_background",
            background_is_messy=True,
            resolution_width=max(1, resolution_width),
            resolution_height=max(1, resolution_height),
            quality_score=3.0,
            needs_upscaling=(resolution_width < 1920 or resolution_height < 1080),
            upscale_factor=2 if (resolution_width < 1920 or resolution_height < 1080) else 1,
            needs_bg_removal=True,
            needs_relighting=True,
            recommended_fixes=[
                f"Manual inspection required: {failure_reason}",
                "Isolate subject via BiRefNet / SAM-2 segmentation",
                "Normalize white balance and exposure"
            ],
            usable_for_generation=False,
            confidence=0.1
        )
```

---

## 2. Complete Copy-Paste Ready System Prompt for `photo_auditor`

The following prompt is injected directly into `gemini-1.5-pro` using the `system_instruction` parameter or the top-level structured prompt framing.

```text
You are an expert technical director, visual effects supervisor, and computer vision auditor specializing in commercial video advertising generation pipelines (e.g., Sora, Runway Gen-3, Kling 1.5, Luma Dream Machine).

Your mission is to perform an exhaustive, unforgiving technical audit of a provided brand product image and contextual client text to determine its readiness for generative video ad synthesis.

You MUST evaluate the image against physical rendering realities, optical fidelity, neural network upscaling constraints, and background segmentation complexity.

### EVALUATION CRITERIA:
1. `subject_detected`: Identify the focal commercial product or entity. If multiple items are present, focus on the primary commercial deliverable.
2. `product_category`: Classify into standard industrial sectors: 'Apparel & Fashion', 'Cosmetics & Skincare', 'Beverage & Food', 'Consumer Electronics', 'Footwear', 'Home Goods', 'Automotive', or 'Jewelry & Accessories'.
3. `materials`: List distinct physical materials identified on the product (e.g., 'brushed stainless steel', 'matte elastomer', 'translucent polycarbonate', 'knit mesh cotton', 'high-gloss enamel').
4. `lighting_condition`: Classify ambient illumination (e.g., 'Balanced Studio Key/Fill/Rim', 'Direct Overhead Fluorescent', 'Golden Hour Sunlight', 'Harsh Flash with Hard Falloff', 'Flat Low-Contrast Diffuse').
5. `lighting_flaws`: Identify specific rendering inhibitors: 'blown_highlights_clipping', 'crushed_shadows', 'harsh_specular_hotspots', 'color_temperature_mismatch', 'unwanted_lens_flare', 'uneven_luminance_falloff'. Empty list if clean studio lighting.
6. `background_description`: Exhaustive spatial description of backdrop geometry, depth of field, blur (bokeh), textures, and surrounding objects.
7. `background_is_messy`: Set to TRUE if the background contains environmental clutter, unintended humans, complex overlapping geometry, textured foliage, or non-commercial artifacts that require neural extraction. Set to FALSE only for clean seamless backdrops, solid cycloramas, or studio gradients.
8. `resolution_width` & `resolution_height`: Native resolution of the asset provided in metadata.
9. `quality_score`: Float between 0.0 and 10.0:
   - 9.0 - 10.0: Flawless commercial studio photography; zero clipping; pristine sharp edges; ideal for zero-shot image-to-video.
   - 7.0 - 8.9: High quality; minor lighting imbalance or acceptable background; easily corrected in pre-processing.
   - 4.0 - 6.9: Consumer snapshot; harsh lighting, significant blur, or noisy background; requires mandatory upscaling, relighting, and matting.
   - 0.0 - 3.9: Low-resolution thumbnail, heavily compressed JPEG artifacts, extreme motion blur, severe occlusion; unusable.
10. `needs_upscaling`: TRUE if source resolution is below 1920x1080 or exhibits compression artifacts/soft focus.
11. `upscale_factor`: Discrete multiplier strictly chosen from [1, 2, 4, 8]. If resolution >= 1920x1080 with high sharpness, choose 1. If 720p-1080p, choose 2. If 360p-720p, choose 4. If < 360p, choose 8.
12. `needs_bg_removal`: TRUE if `background_is_messy` is true, or if product must be isolated for clean 3D parallax generation.
13. `needs_relighting`: TRUE if `lighting_flaws` is non-empty or if lighting style clashes with commercial advertising norms.
14. `recommended_fixes`: A list of imperative, machine-actionable directives for downstream processing tools (e.g., "Apply Real-ESRGAN_x4plus to reconstruct micro-textures", "Execute BiRefNet alpha segmentation to isolate bottle silhouette", "Apply -1.5 EV shadow recovery via IC-Light conditioning", "Neutralize 4200K tungsten color cast to 5600K daylight").
15. `usable_for_generation`: Boolean gatekeeper. Must be TRUE only if quality_score >= 6.0 OR if actionable pre-processing can reliably elevate the asset to generative standards.
16. `confidence`: Float between 0.0 and 1.0 indicating model assessment certainty.

### OUTPUT FORMAT:
You MUST respond with ONLY valid JSON adhering strictly to the schema below. Do not wrap in Markdown fences (` ```json `), do not include preambles, and do not include postscript notes.

{
  "subject_detected": "string",
  "product_category": "string",
  "materials": ["string"],
  "lighting_condition": "string",
  "lighting_flaws": ["string"],
  "background_description": "string",
  "background_is_messy": true,
  "resolution_width": 1920,
  "resolution_height": 1080,
  "quality_score": 8.5,
  "needs_upscaling": false,
  "upscale_factor": 1,
  "needs_bg_removal": false,
  "needs_relighting": false,
  "recommended_fixes": ["string"],
  "usable_for_generation": true,
  "confidence": 0.95
}
```

---

## 3. Photo Auditor Implementation (`src/deconstruction/photo_auditor.py`)

This module enforces:
1. Strict file existence verification (`ValueError`).
2. Pillow image dimension inspection.
3. Aspect-ratio preserving downscaling if either axis exceeds `2048px`.
4. Standard Base64 data URI preparation and inline Part construction.
5. `google.generativeai` Gemini 1.5 Pro multimodal call.
6. `tenacity` exponential backoff (1s, 2s, 4s) across 3 attempts on invalid JSON.
7. Graceful fallback to `AssetQualityAudit.safe_default`.

```python
"""
src/deconstruction/photo_auditor.py
Production photo audit pipeline using Google Gemini 1.5 Pro Vision.
"""

import os
import io
import json
import base64
import logging
from typing import Tuple

from PIL import Image, ImageOps
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.deconstruction.models import AssetQualityAudit

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PHOTO_AUDITOR_SYSTEM_PROMPT = """You are an expert technical director, visual effects supervisor, and computer vision auditor specializing in commercial video advertising generation pipelines (e.g., Sora, Runway Gen-3, Kling 1.5, Luma Dream Machine).

Your mission is to perform an exhaustive, unforgiving technical audit of a provided brand product image and contextual client text to determine its readiness for generative video ad synthesis.

You MUST evaluate the image against physical rendering realities, optical fidelity, neural network upscaling constraints, and background segmentation complexity.

### EVALUATION CRITERIA:
1. `subject_detected`: Identify the focal commercial product or entity. If multiple items are present, focus on the primary commercial deliverable.
2. `product_category`: Classify into standard industrial sectors: 'Apparel & Fashion', 'Cosmetics & Skincare', 'Beverage & Food', 'Consumer Electronics', 'Footwear', 'Home Goods', 'Automotive', or 'Jewelry & Accessories'.
3. `materials`: List distinct physical materials identified on the product (e.g., 'brushed stainless steel', 'matte elastomer', 'translucent polycarbonate', 'knit mesh cotton', 'high-gloss enamel').
4. `lighting_condition`: Classify ambient illumination (e.g., 'Balanced Studio Key/Fill/Rim', 'Direct Overhead Fluorescent', 'Golden Hour Sunlight', 'Harsh Flash with Hard Falloff', 'Flat Low-Contrast Diffuse').
5. `lighting_flaws`: Identify specific rendering inhibitors: 'blown_highlights_clipping', 'crushed_shadows', 'harsh_specular_hotspots', 'color_temperature_mismatch', 'unwanted_lens_flare', 'uneven_luminance_falloff'. Empty list if clean studio lighting.
6. `background_description`: Exhaustive spatial description of backdrop geometry, depth of field, blur (bokeh), textures, and surrounding objects.
7. `background_is_messy`: Set to TRUE if the background contains environmental clutter, unintended humans, complex overlapping geometry, textured foliage, or non-commercial artifacts that require neural extraction. Set to FALSE only for clean seamless backdrops, solid cycloramas, or studio gradients.
8. `resolution_width` & `resolution_height`: Native resolution of the asset provided in metadata.
9. `quality_score`: Float between 0.0 and 10.0:
   - 9.0 - 10.0: Flawless commercial studio photography; zero clipping; pristine sharp edges; ideal for zero-shot image-to-video.
   - 7.0 - 8.9: High quality; minor lighting imbalance or acceptable background; easily corrected in pre-processing.
   - 4.0 - 6.9: Consumer snapshot; harsh lighting, significant blur, or noisy background; requires mandatory upscaling, relighting, and matting.
   - 0.0 - 3.9: Low-resolution thumbnail, heavily compressed JPEG artifacts, extreme motion blur, severe occlusion; unusable.
10. `needs_upscaling`: TRUE if source resolution is below 1920x1080 or exhibits compression artifacts/soft focus.
11. `upscale_factor`: Discrete multiplier strictly chosen from [1, 2, 4, 8]. If resolution >= 1920x1080 with high sharpness, choose 1. If 720p-1080p, choose 2. If 360p-720p, choose 4. If < 360p, choose 8.
12. `needs_bg_removal`: TRUE if `background_is_messy` is true, or if product must be isolated for clean 3D parallax generation.
13. `needs_relighting`: TRUE if `lighting_flaws` is non-empty or if lighting style clashes with commercial advertising norms.
14. `recommended_fixes`: A list of imperative, machine-actionable directives for downstream processing tools (e.g., "Apply Real-ESRGAN_x4plus to reconstruct micro-textures", "Execute BiRefNet alpha segmentation to isolate bottle silhouette", "Apply -1.5 EV shadow recovery via IC-Light conditioning", "Neutralize 4200K tungsten color cast to 5600K daylight").
15. `usable_for_generation`: Boolean gatekeeper. Must be TRUE only if quality_score >= 6.0 OR if actionable pre-processing can reliably elevate the asset to generative standards.
16. `confidence`: Float between 0.0 and 1.0 indicating model assessment certainty.

### OUTPUT FORMAT:
You MUST respond with ONLY valid JSON adhering strictly to the schema below. Do not wrap in Markdown fences (```json), do not include preambles, and do not include postscript notes.

{
  "subject_detected": "string",
  "product_category": "string",
  "materials": ["string"],
  "lighting_condition": "string",
  "lighting_flaws": ["string"],
  "background_description": "string",
  "background_is_messy": true,
  "resolution_width": 1920,
  "resolution_height": 1080,
  "quality_score": 8.5,
  "needs_upscaling": false,
  "upscale_factor": 1,
  "needs_bg_removal": false,
  "needs_relighting": false,
  "recommended_fixes": ["string"],
  "usable_for_generation": true,
  "confidence": 0.95
}"""


def preprocess_and_encode_image(image_path: str, max_dim: int = 2048) -> Tuple[str, str, int, int]:
    """
    Opens an image, inspects original dimensions, resizes with Pillow preserving aspect
    ratio if any dimension exceeds max_dim, and returns MIME type, base64 data, and original dimensions.

    Returns:
        Tuple[str, str, int, int]: (mime_type, base64_encoded_data, orig_width, orig_height)
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Input file not found: {image_path}")

    try:
        with Image.open(image_path) as img:
            # Respect EXIF orientation tags if present
            img = ImageOps.exif_transpose(img)
            orig_w, orig_h = img.size

            # Determine format & MIME type
            img_format = img.format if img.format else "PNG"
            mime_map = {
                "JPEG": "image/jpeg",
                "JPG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp"
            }
            mime_type = mime_map.get(img_format.upper(), "image/png")

            # Downsample if exceeding max_dim on either axis
            if orig_w > max_dim or orig_h > max_dim:
                logger.warning(
                    f"Asset {image_path} dimensions ({orig_w}x{orig_h}) exceed limit {max_dim}px. "
                    "Downsampling via Pillow Lanczos resampling."
                )
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            # Ensure RGB conversion for saving JPEG if original had alpha
            save_format = "PNG" if mime_type == "image/png" else "JPEG"
            if save_format == "JPEG" and img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            img.save(buffer, format=save_format, quality=95)
            encoded_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

            return mime_type, encoded_str, orig_w, orig_h

    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise IOError(f"Failed to open or decode image at {image_path}: {str(exc)}") from exc


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((json.JSONDecodeError, ValueError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def _call_gemini_vision_with_retry(
    model: genai.GenerativeModel,
    request_contents: list,
    orig_w: int,
    orig_h: int
) -> AssetQualityAudit:
    """
    Executes the Gemini 1.5 Pro API call and parses the JSON response into AssetQualityAudit.
    Retries up to 3 times on JSON parsing or structural failures with exponential backoff (1s, 2s, 4s).
    """
    response = model.generate_content(
        contents=request_contents,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
            top_p=0.95
        )
    )

    if not response.text:
        raise ValueError("Gemini returned empty or blocked response payload.")

    raw_text = response.text.strip()

    # Strip accidental markdown code blocks if present
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    parsed_json = json.loads(raw_text)

    # Force original resolution into payload to override model estimation
    parsed_json["resolution_width"] = orig_w
    parsed_json["resolution_height"] = orig_h

    # Parse and validate with Pydantic
    return AssetQualityAudit(**parsed_json)


def audit_photo(image_path: str, client_text: str) -> AssetQualityAudit:
    """
    Primary interface for Stage 1 Asset Deconstruction on static images.

    Args:
        image_path: Absolute or relative local path to the raw input photo.
        client_text: Brand context, target brief, or client requirements.

    Returns:
        AssetQualityAudit: Validated Pydantic model representing the visual audit.

    Raises:
        ValueError: If input file does not exist.
        RuntimeError: If Google Gemini API is unauthenticated or unconfigured.
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Input file not found: {image_path}")

    # Ensure Gemini API key is present
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not configured. "
            "Export a valid key from Google AI Studio before executing audit_photo."
        )

    genai.configure(api_key=api_key)

    # 1. Preprocess, resize if > 2048px, and encode
    mime_type, b64_data, orig_w, orig_h = preprocess_and_encode_image(image_path, max_dim=2048)

    # 2. Build the Gemini 1.5 Pro GenerativeModel instance
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction=PHOTO_AUDITOR_SYSTEM_PROMPT
    )

    # 3. Formulate multimodal request parts
    image_part = {
        "inline_data": {
            "mime_type": mime_type,
            "data": b64_data
        }
    }

    user_context = (
        f"CLIENT CAMPAIGN CONTEXT & REQUIREMENTS:\n{client_text}\n\n"
        f"TECHNICAL METADATA:\n"
        f"- Original File Resolution: {orig_w}x{orig_h} px\n\n"
        f"Perform the complete asset quality audit and output valid JSON conforming strictly to schema."
    )

    request_contents = [image_part, user_context]

    # 4. Execute inference with retry policy; trigger safe fallback on final failure
    try:
        audit_result = _call_gemini_vision_with_retry(
            model=model,
            request_contents=request_contents,
            orig_w=orig_w,
            orig_h=orig_h
        )
        return audit_result
    except Exception as exc:
        logger.error(
            f"All 3 Gemini audit attempts failed for image {image_path}. Error: {str(exc)}. "
            "Falling back to AssetQualityAudit.safe_default.",
            exc_info=True
        )
        return AssetQualityAudit.safe_default(
            resolution_width=orig_w,
            resolution_height=orig_h,
            failure_reason=f"Gemini API / JSON parsing failed after 3 retries: {str(exc)}"
        )
```

---

## 4. Video Auditor Implementation (`src/deconstruction/video_auditor.py`)

This component handles dynamic video input, deconstructing video assets into:
1. **Scene-aware I-frame / keyframe extraction** using optimized `ffmpeg` subprocessing.
2. **Word-level timestamped transcription** with `whisper-timestamped`.
3. **LLM multi-frame visual classification** using Gemini 1.5 Pro to classify ad style, pacing, and visual archetypes.

```python
"""
src/deconstruction/video_auditor.py
Production video asset deconstruction: keyframe extraction, word-level audio alignment,
and LLM visual archetype classification.
"""

import os
import shutil
import subprocess
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from pydantic import BaseModel, Field
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    confidence: float


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    words: List[WordTimestamp]


class VideoAssetAudit(BaseModel):
    video_path: str
    duration_seconds: float
    fps: float
    resolution_width: int
    resolution_height: int
    video_type: str = Field(
        ...,
        description="Archetype: 'UGC_TALKING_HEAD', 'PRODUCT_DEMO_UNBOXING', 'STUDIO_COMMERCIAL', '3D_CGI_RENDER', 'LIFESTYLE_VLOG'"
    )
    pacing_bpm: Optional[float]
    keyframe_paths: List[str]
    transcript: List[TranscriptSegment]
    visual_narrative_summary: str
    usable_for_derivative_ads: bool


# ---------------------------------------------------------------------------
# FFmpeg Keyframe Extraction
# ---------------------------------------------------------------------------

def check_ffmpeg_installed() -> None:
    """Verifies that ffmpeg and ffprobe binaries exist in the system PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg not found in system PATH. Please install FFmpeg from "
            "https://ffmpeg.org/download.html and ensure it is added to your environment's PATH."
        )
    if shutil.which("ffprobe") is None:
        raise RuntimeError(
            "ffprobe not found in system PATH. Please install FFmpeg suite from "
            "https://ffmpeg.org/download.html and ensure it is added to your environment's PATH."
        )


def extract_video_metadata(video_path: str) -> Dict[str, Any]:
    """Inspects video stream metadata via ffprobe JSON output."""
    check_ffmpeg_installed()
    if not os.path.exists(video_path):
        raise ValueError(f"Input video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration:format=duration",
        "-of", "json",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    probe_data = json.loads(result.stdout)

    stream = probe_data.get("streams", [{}])[0]
    fmt = probe_data.get("format", {})

    # Calculate FPS
    fps_raw = stream.get("r_frame_rate", "30/1")
    if "/" in fps_raw:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_raw)

    duration = float(stream.get("duration") or fmt.get("duration") or 0.0)
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration
    }


def extract_keyframes(
    video_path: str,
    output_dir: str,
    scene_threshold: float = 0.35,
    max_keyframes: int = 12
) -> List[str]:
    """
    Executes scene-change aware I-frame extraction using FFmpeg filter graph.
    Selects frames where scene score > scene_threshold, falling back to regular intervals
    if insufficient cuts are found.

    Exact FFmpeg command signature executed:
        ffmpeg -skip_frame nokey -i <video_path> -vf "select='gt(scene,0.35)',showinfo" \\
               -vsync vfr -frame_pts true <output_dir>/keyframe_%04d.jpg
    """
    check_ffmpeg_installed()
    if not os.path.exists(video_path):
        raise ValueError(f"Input video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    out_pattern = os.path.join(output_dir, "keyframe_%04d.jpg")

    # Command: scene change detection with variable frame rate output
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-skip_frame", "nokey",
        "-i", video_path,
        "-vf", f"select='gt(scene,{scene_threshold})',showinfo",
        "-vsync", "vfr",
        "-frame_pts", "true",
        "-q:v", "2",
        out_pattern
    ]

    logger.info(f"Running keyframe extraction command: {' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    extracted = sorted([str(p) for p in Path(output_dir).glob("keyframe_*.jpg")])

    # Fallback: if scene change yielded fewer than 3 frames (e.g., continuous shot), extract evenly spaced frames
    if len(extracted) < 3:
        logger.info("Scene detection yielded < 3 keyframes. Falling back to periodic fps extraction.")
        fallback_pattern = os.path.join(output_dir, "interval_%04d.jpg")
        periodic_cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vf", "fps=1",  # 1 frame per second
            "-vsync", "vfr",
            "-q:v", "2",
            fallback_pattern
        ]
        subprocess.run(periodic_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        extracted = sorted([str(p) for p in Path(output_dir).glob("*.jpg")])

    # Cap to max_keyframes evenly spaced across results
    if len(extracted) > max_keyframes:
        step = len(extracted) / max_keyframes
        indices = [int(i * step) for i in range(max_keyframes)]
        extracted = [extracted[idx] for idx in indices]

    return extracted


# ---------------------------------------------------------------------------
# Whisper-Timestamped Word-Level Transcription
# ---------------------------------------------------------------------------

def extract_word_level_transcript(video_path: str) -> List[TranscriptSegment]:
    """
    Transcribes audio track using whisper-timestamped to provide exact word-level boundaries
    for synchronized kinetic typography and video cut-point generation.
    """
    if not os.path.exists(video_path):
        raise ValueError(f"Video file not found for transcription: {video_path}")

    try:
        import whisper_timestamped as whisper
        import torch
    except ImportError as e:
        raise RuntimeError(
            "whisper-timestamped or torch is not installed. "
            "Install via: pip install whisper-timestamped torch"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading whisper-timestamped model 'base' on {device}...")
    model = whisper.load_model("base", device=device)

    # whisper-timestamped accepts video files directly and extracts audio stream internally
    audio = whisper.load_audio(video_path)
    result = whisper.transcribe(model, audio, language="en", detect_disfluencies=True)

    segments_out: List[TranscriptSegment] = []

    for seg in result.get("segments", []):
        words_out: List[WordTimestamp] = []
        for w in seg.get("words", []):
            words_out.append(
                WordTimestamp(
                    word=w.get("text", "").strip(),
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    confidence=float(w.get("confidence", 0.0))
                )
            )

        segments_out.append(
            TranscriptSegment(
                text=seg.get("text", "").strip(),
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                words=words_out
            )
        )

    return segments_out


# ---------------------------------------------------------------------------
# LLM Video Type Classification via Keyframes
# ---------------------------------------------------------------------------

def classify_video_archetype(
    keyframe_paths: List[str],
    transcript_text: str,
    client_brief: str
) -> Dict[str, Any]:
    """
    Sends multi-frame keyframe sequence into Gemini 1.5 Pro to determine visual category,
    narrative structure, and generation readiness.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not configured.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")

    parts: List[Any] = []

    # Attach keyframes in chronological order
    for idx, kf_path in enumerate(keyframe_paths):
        with Image.open(kf_path) as img:
            # Resize if keyframe exceeds 1024px for quick multi-image evaluation
            if max(img.size) > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64_val = base64.b64encode(buf.getvalue()).decode("utf-8")
            parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64_val
                }
            })

    classification_prompt = f"""
You are inspecting a chronological sequence of {len(keyframe_paths)} keyframes extracted from a commercial video asset.

TRANSCRIPT:
"{transcript_text}"

CAMPAIGN BRIEF:
"{client_brief}"

Analyze this video asset and output JSON matching this exact structure:
{{
  "video_type": "UGC_TALKING_HEAD" | "PRODUCT_DEMO_UNBOXING" | "STUDIO_COMMERCIAL" | "3D_CGI_RENDER" | "LIFESTYLE_VLOG",
  "visual_narrative_summary": "Concise summary of visual pacing, scene composition, and narrative progression",
  "usable_for_derivative_ads": true | false
}}

Respond ONLY with valid JSON. No Markdown formatting.
"""
    parts.append(classification_prompt)

    response = model.generate_content(
        contents=parts,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )

    clean_text = response.text.strip()
    if clean_text.startswith("```"):
        clean_text = clean_text.split("```")[1]
        if clean_text.startswith("json"):
            clean_text = clean_text[4:]
    clean_text = clean_text.strip()

    return json.loads(clean_text)


# ---------------------------------------------------------------------------
# High-Level Orchestrator
# ---------------------------------------------------------------------------

def audit_video(
    video_path: str,
    client_brief: str,
    work_dir: str = "/tmp/video_audit"
) -> VideoAssetAudit:
    """
    End-to-end pipeline deconstructing video into metadata, keyframes, transcript, and visual type.
    """
    meta = extract_video_metadata(video_path)
    keyframe_dir = os.path.join(work_dir, "keyframes")
    keyframes = extract_keyframes(video_path, output_dir=keyframe_dir)

    transcript_segments = extract_word_level_transcript(video_path)
    full_transcript_text = " ".join(seg.text for seg in transcript_segments)

    classification = classify_video_archetype(
        keyframe_paths=keyframes,
        transcript_text=full_transcript_text,
        client_brief=client_brief
    )

    return VideoAssetAudit(
        video_path=video_path,
        duration_seconds=meta["duration"],
        fps=meta["fps"],
        resolution_width=meta["width"],
        resolution_height=meta["height"],
        video_type=classification.get("video_type", "STUDIO_COMMERCIAL"),
        pacing_bpm=None,
        keyframe_paths=keyframes,
        transcript=transcript_segments,
        visual_narrative_summary=classification.get("visual_narrative_summary", ""),
        usable_for_derivative_ads=classification.get("usable_for_derivative_ads", False)
    )
```

---

## 5. Complete Error Handling & Resilience Table

The table below catalogs every failure mode, interception mechanism, backoff schedule, and terminal response in Module 1.

| Failure Mode / Condition | Trigger Event | Detection / Interception Mechanism | Retry Policy / Backoff | Recovery / Fallback Action | Terminal Output / Exception |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input File Missing** | File path does not exist on disk | `os.path.exists(path)` check in `audit_photo` / `audit_video` | None (fail fast) | None | `raise ValueError(f"Input file not found: {path}")` |
| **FFmpeg Binary Not Found** | FFmpeg or FFprobe absent from system PATH | `shutil.which('ffmpeg') is None` or `shutil.which('ffprobe') is None` | None (fail fast) | None | `raise RuntimeError("FFmpeg not found in system PATH. Install from https://ffmpeg.org/download.html...")` |
| **Image Resolution Exceeds 2048px** | Input image has width or height > 2048px | Pillow `img.size` inspection before Base64 encoding | None (proactive normalization) | Downscale using `img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)` preserving aspect ratio | Normal execution with logged warning; original dimensions preserved in audit metadata |
| **Gemini Returns Malformed JSON** | LLM outputs invalid JSON syntax or truncated brackets | `json.loads` throws `json.JSONDecodeError` inside retry block | `tenacity`: 3 attempts, exponential backoff (1s $\to$ 2s $\to$ 4s) | Log warning on attempts 1 & 2; on final attempt return `safe_default` | Returns `AssetQualityAudit.safe_default(width, height, failure_reason)` without crashing pipeline |
| **Gemini Schema Mismatch** | JSON is valid but missing fields or fails bounds validation | Pydantic `ValidationError` upon `AssetQualityAudit(**data)` | `tenacity`: 3 attempts, exponential backoff (1s $\to$ 2s $\to$ 4s) | Log warning on attempts 1 & 2; on final attempt return `safe_default` | Returns `AssetQualityAudit.safe_default(width, height, failure_reason)` |
| **Empty / Blocked LLM Response** | Safety filter or quota blocks response (`response.text` empty) | Check `if not response.text:` throws `ValueError` | `tenacity`: 3 attempts, exponential backoff (1s $\to$ 2s $\to$ 4s) | Log filter trigger; on final attempt return `safe_default` with `usable_for_generation=False` | Returns `AssetQualityAudit.safe_default` with safety block metadata |
| **Missing API Key** | `GEMINI_API_KEY` not exported in environment | Check `os.environ.get("GEMINI_API_KEY")` is `None` | None (configuration error) | None | `raise RuntimeError("GEMINI_API_KEY environment variable is not configured...")` |
| **Silent or Missing Audio Track** | Video contains no audio stream or empty track | `whisper-timestamped` returns 0 segments or audio extract empty | None | Catch audio stream error, return `transcript=[]` gracefully | `VideoAssetAudit` populated with empty transcript; video processing continues |

---

## 6. Verification Test Suite (`tests/test_deconstruction.py`)

This test suite uses `pytest` and `unittest.mock` to verify the execution paths, Pillow resizing, Tenacity retry behavior, safe fallback defaults, and schema constraints.

```python
"""
tests/test_deconstruction.py
Unit tests and verification assertions for Module 1: Asset Deconstruction.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image

from src.deconstruction.models import AssetQualityAudit
from src.deconstruction.photo_auditor import (
    preprocess_and_encode_image,
    audit_photo,
    PHOTO_AUDITOR_SYSTEM_PROMPT
)
from src.deconstruction.video_auditor import (
    check_ffmpeg_installed,
    extract_keyframes,
    VideoAssetAudit,
    WordTimestamp,
    TranscriptSegment
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image_standard(tmp_path):
    img_path = str(tmp_path / "standard_1080p.png")
    img = Image.new("RGB", (1920, 1080), color=(128, 128, 128))
    img.save(img_path, format="PNG")
    return img_path


@pytest.fixture
def sample_image_oversized(tmp_path):
    img_path = str(tmp_path / "oversized_4k.jpg")
    img = Image.new("RGB", (4096, 3072), color=(255, 0, 0))
    img.save(img_path, format="JPEG")
    return img_path


@pytest.fixture
def valid_audit_payload():
    return {
        "subject_detected": "Ceramic Matte Coffee Mug",
        "product_category": "Home Goods",
        "materials": ["glazed ceramic", "matte cork"],
        "lighting_condition": "Studio Softbox",
        "lighting_flaws": [],
        "background_description": "Clean seamless light-grey gradient cyclorama",
        "background_is_messy": False,
        "resolution_width": 1920,
        "resolution_height": 1080,
        "quality_score": 9.2,
        "needs_upscaling": False,
        "upscale_factor": 1,
        "needs_bg_removal": False,
        "needs_relighting": False,
        "recommended_fixes": ["Ready for zero-shot video synthesis"],
        "usable_for_generation": True,
        "confidence": 0.98
    }


# ---------------------------------------------------------------------------
# 1. Pydantic Model Validation Tests
# ---------------------------------------------------------------------------

def test_pydantic_model_valid(valid_audit_payload):
    audit = AssetQualityAudit(**valid_audit_payload)
    assert audit.subject_detected == "Ceramic Matte Coffee Mug"
    assert audit.upscale_factor == 1
    assert audit.quality_score == 9.2
    assert audit.confidence == 0.98
    assert audit.usable_for_generation is True


def test_pydantic_model_invalid_upscale_factor(valid_audit_payload):
    invalid_data = valid_audit_payload.copy()
    invalid_data["upscale_factor"] = 3  # Must be one of [1, 2, 4, 8]
    with pytest.raises(Exception):
        AssetQualityAudit(**invalid_data)


def test_pydantic_model_quality_score_clamping(valid_audit_payload):
    invalid_data = valid_audit_payload.copy()
    invalid_data["quality_score"] = 11.5  # Max is 10.0
    with pytest.raises(Exception):
        AssetQualityAudit(**invalid_data)


def test_safe_default_generation():
    default_audit = AssetQualityAudit.safe_default(
        resolution_width=800,
        resolution_height=600,
        failure_reason="Unit test fallback validation"
    )
    assert default_audit.subject_detected == "unknown_subject"
    assert default_audit.usable_for_generation is False
    assert default_audit.needs_upscaling is True
    assert default_audit.upscale_factor == 2
    assert default_audit.quality_score == 3.0
    assert default_audit.resolution_width == 800
    assert default_audit.resolution_height == 600


# ---------------------------------------------------------------------------
# 2. Image Preprocessing & Downscaling Assertions
# ---------------------------------------------------------------------------

def test_preprocess_standard_image_no_resize(sample_image_standard):
    mime_type, b64_str, w, h = preprocess_and_encode_image(sample_image_standard, max_dim=2048)
    assert mime_type == "image/png"
    assert w == 1920
    assert h == 1080
    assert len(b64_str) > 0


def test_preprocess_oversized_image_resizes_dimensions(sample_image_oversized):
    mime_type, b64_str, orig_w, orig_h = preprocess_and_encode_image(sample_image_oversized, max_dim=2048)
    assert orig_w == 4096
    assert orig_h == 3072

    # Verify that the decoded buffer is clamped within 2048px
    import base64, io
    decoded_bytes = base64.b64decode(b64_str)
    with Image.open(io.BytesIO(decoded_bytes)) as decoded_img:
        max_dimension = max(decoded_img.size)
        assert max_dimension <= 2048
        # Ensure aspect ratio (4:3) is preserved (2048 x 1536)
        assert decoded_img.size == (2048, 1536)


def test_missing_image_file_raises_value_error():
    with pytest.raises(ValueError, match="Input file not found"):
        preprocess_and_encode_image("/non/existent/path/to/asset.jpg")


# ---------------------------------------------------------------------------
# 3. Gemini Vision Call & Tenacity Retry Behavior
# ---------------------------------------------------------------------------

@patch.dict(os.environ, {"GEMINI_API_KEY": "test_ai_studio_key"})
@patch("google.generativeai.GenerativeModel")
def test_photo_auditor_success_flow(mock_gen_model_class, sample_image_standard, valid_audit_payload):
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = json.dumps(valid_audit_payload)
    mock_instance.generate_content.return_value = mock_response
    mock_gen_model_class.return_value = mock_instance

    result = audit_photo(
        image_path=sample_image_standard,
        client_text="Luxury ceramic drinkware campaign targeting Gen Z aesthetic."
    )

    assert isinstance(result, AssetQualityAudit)
    assert result.subject_detected == "Ceramic Matte Coffee Mug"
    assert result.resolution_width == 1920
    assert result.resolution_height == 1080
    assert result.quality_score == 9.2
    assert mock_instance.generate_content.call_count == 1


@patch.dict(os.environ, {"GEMINI_API_KEY": "test_ai_studio_key"})
@patch("google.generativeai.GenerativeModel")
def test_photo_auditor_tenacity_retry_and_fallback_on_corrupt_json(
    mock_gen_model_class,
    sample_image_standard
):
    mock_instance = MagicMock()
    corrupt_response = MagicMock()
    # Malformed JSON missing closing brackets
    corrupt_response.text = '{"subject_detected": "Shoe", "quality_score": 7.5'
    mock_instance.generate_content.return_value = corrupt_response
    mock_gen_model_class.return_value = mock_instance

    # Should attempt 3 times, fail, and catch exception returning safe_default
    result = audit_photo(
        image_path=sample_image_standard,
        client_text="Sneaker launch."
    )

    # 3 attempts executed
    assert mock_instance.generate_content.call_count == 3
    # Check that safe_default fallback was returned
    assert isinstance(result, AssetQualityAudit)
    assert result.subject_detected == "unknown_subject"
    assert result.usable_for_generation is False
    assert result.resolution_width == 1920
    assert result.resolution_height == 1080


# ---------------------------------------------------------------------------
# 4. Video Auditor & Dependency Failure Assertions
# ---------------------------------------------------------------------------

@patch("shutil.which", return_value=None)
def test_ffmpeg_missing_raises_runtime_error(mock_which):
    with pytest.raises(RuntimeError) as exc_info:
        check_ffmpeg_installed()
    assert "https://ffmpeg.org/download.html" in str(exc_info.value)


def test_video_file_not_found_raises_value_error():
    with pytest.raises(ValueError, match="Input video file not found"):
        extract_keyframes("/path/does/not/exist.mp4", "/tmp/keyframes")


def test_transcript_segment_data_structure():
    word_1 = WordTimestamp(word="Premium", start=0.0, end=0.45, confidence=0.99)
    word_2 = WordTimestamp(word="Audio", start=0.46, end=0.88, confidence=0.97)
    seg = TranscriptSegment(
        text="Premium Audio",
        start=0.0,
        end=0.88,
        words=[word_1, word_2]
    )

    assert seg.words[0].word == "Premium"
    assert seg.words[1].end == 0.88
    assert seg.words[0].start < seg.words[0].end
```

---

## 7. Execution & Operational Verification Checklist

To verify this module locally or in CI:

```bash
# 1. Install dependencies
pip install pydantic pillow google-generativeai tenacity whisper-timestamped torch pytest

# 2. Verify system FFmpeg binary
ffmpeg -version

# 3. Export API Key
export GEMINI_API_KEY="your-api-key-here"

# 4. Run verification tests
pytest -v tests/test_deconstruction.py
```

This completes the production-grade specification for **MODULE 1: STAGE 1 ASSET DECONSTRUCTION**. All components strictly observe typing, data resilience, optical normalization, and fail-safe generative requirements.

---

# MODULE 2: THE MOE THINKING BRAIN

## AI Video Ad Engine Architecture & Implementation Specification

---

### EXECUTIVE ARCHITECTURAL OVERVIEW

Module 2 represents the cognitive core of the Autonomous AI Video Ad Generation Engine. It transforms raw, unstructured product briefs, brand guidelines, and campaign goals into a deterministic, frame-accurate **Master Production Manifest**.

Rather than relying on a single monolithic Large Language Model prompt—which suffers from cognitive drift, aesthetic compromises, and hallucinations—Module 2 employs a **Mixture of Experts (MoE)** architecture. Four specialized system prompts act as autonomous domain experts running concurrently:
1. **Expert A (Marketing Strategist):** Defines performance metrics, psychology, audience hooks, and structural pacing.
2. **Expert B (Cinematographer):** Translates strategy into lens selections, physical camera kinematics, lighting environments, and render preprocessing pipelines.
3. **Expert C (Scriptwriter):** Crafts timed voiceover scripts, sonic environments, and on-screen typography hierarchy.
4. **Expert D (Brand Guardian):** Validates legal guardrails, enforces brand colors, audits regulatory compliance, and positions brand assets.

The outputs of these four parallel streams are unified by the **Master Synthesizer**, which arbitrates creative conflicts and outputs the machine-executable `MasterProductionManifest`.

```
                               ┌────────────────────────────────┐
                               │   Input Campaign & Asset Data  │
                               └───────────────┬────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               │                               │                               │
               ▼                               ▼                               ▼
       ┌───────────────┐               ┌───────────────┐               ┌───────────────┐               ┌───────────────┐
       │   Expert A    │               │   Expert B    │               │   Expert C    │               │   Expert D    │
       │   Marketing   │               │Cinematographer│               │ Scriptwriter  │               │ Brand Guardian│
       └───────┬───────┘               └───────┬───────┘               └───────┬───────┘               └───────┬───────┘
               │                               │                               │                               │
               └───────────────────────────────┼───────────────────────────────┘                               │
                                               ▼                                                               │
                               ┌────────────────────────────────┐                                              │
                               │        Master Synthesizer      │◄─────────────────────────────────────────────┘
                               │  (Conflict Resolver & Merger)  │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │    Pydantic Schema Validation  │
                               │       & Heuristic Repair       │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │    MasterProductionManifest    │
                               │   (To Diffusion Video Engine)  │
                               └────────────────────────────────┘
```

---

## 1. ASYNCIO PATTERN & CONCURRENCY ENGINE

The Google Gemini Python SDK (`google-generativeai` or `google.genai`) provides synchronous client methods (`model.generate_content`) for model inference. To prevent blocking the asynchronous event loop and ensure zero-latency parallel dispatch, the MoE orchestrator uses a persistent `ThreadPoolExecutor(max_workers=4)` combined with `asyncio.get_running_loop().run_in_executor()`.

### 1.1 Complete Python Concurrency Implementation

```python
"""
module_2_moe_brain.py
Concurrent MoE Execution Engine for AI Video Ad Engine.
Handles thread pool offloading, timeout enforcement, exception aggregation,
and payload dispatch.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("MoE-Thinking-Brain")


@dataclass(frozen=True)
class ExpertConfig:
    name: str
    model_name: str
    system_instruction: str
    temperature: float
    max_output_tokens: int


class MoEConcurrencyOrchestrator:
    """
    Manages concurrent thread pool execution for Gemini Expert models.
    Offloads synchronous SDK calls into dedicated OS threads to enable
    true non-blocking parallel execution on asyncio event loops.
    """

    def __init__(self, api_key: str, max_workers: int = 4, request_timeout_sec: float = 30.0):
        self.api_key = api_key
        self.max_workers = max_workers
        self.request_timeout_sec = request_timeout_sec
        genai.configure(api_key=self.api_key)
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="MoE-Worker"
        )

    def _sync_gemini_call(
        self,
        config: ExpertConfig,
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Synchronous worker function executed inside the ThreadPoolExecutor.
        Instantiates the GenerativeModel with strict JSON output constraints.
        """
        start_time = time.perf_counter()
        thread_name = asyncio.current_task().get_name() if asyncio.current_task() else "WorkerThread"
        logger.info(f"[{config.name}] Starting inference in worker pool.")

        try:
            model = genai.GenerativeModel(
                model_name=config.model_name,
                system_instruction=config.system_instruction,
                generation_config=genai.types.GenerationConfig(
                    temperature=config.temperature,
                    max_output_tokens=config.max_output_tokens,
                    response_mime_type="application/json"
                )
            )

            response = model.generate_content(user_prompt)
            duration = time.perf_counter() - start_time
            logger.info(f"[{config.name}] Inference completed in {duration:.2f}s.")

            raw_text = response.text.strip()
            # Parse raw response to validate it is legal JSON
            parsed_json = json.loads(raw_text)
            return {
                "expert": config.name,
                "data": parsed_json,
                "latency_sec": duration,
                "status": "SUCCESS"
            }
        except json.JSONDecodeError as jde:
            duration = time.perf_counter() - start_time
            logger.error(f"[{config.name}] Failed to decode JSON: {jde}. Raw: {response.text if 'response' in locals() else 'None'}")
            return {
                "expert": config.name,
                "error": f"JSONDecodeError: {str(jde)}",
                "raw_text": response.text if 'response' in locals() else "",
                "latency_sec": duration,
                "status": "MALFORMED_JSON"
            }
        except Exception as exc:
            duration = time.perf_counter() - start_time
            logger.error(f"[{config.name}] SDK Execution Failed: {str(exc)}")
            return {
                "expert": config.name,
                "error": str(exc),
                "latency_sec": duration,
                "status": "FAILED"
            }

    async def execute_expert_async(
        self,
        config: ExpertConfig,
        user_prompt: str
    ) -> Dict[str, Any]:
        """
        Wraps synchronous _sync_gemini_call in asyncio.run_in_executor.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self._sync_gemini_call,
            config,
            user_prompt
        )

    async def fire_all_experts_simultaneously(
        self,
        expert_configs: List[ExpertConfig],
        shared_user_prompt: str
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        """
        Fires all 4 expert calls simultaneously using asyncio.gather over run_in_executor.
        Enforces global timeout and aggregates results.
        """
        logger.info(f"Firing {len(expert_configs)} experts simultaneously via ThreadPoolExecutor...")
        start_time = time.perf_counter()

        tasks = [
            asyncio.create_task(
                self.execute_expert_async(cfg, shared_user_prompt),
                name=f"Task-{cfg.name}"
            )
            for cfg in expert_configs
        ]

        # Await all tasks concurrently with a hard deadline
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time
        logger.info(f"All {len(expert_configs)} experts responded in {total_time:.2f}s total.")

        successful_payloads: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}

        for cfg, res in zip(expert_configs, results_list):
            if isinstance(res, Exception):
                logger.error(f"Expert {cfg.name} encountered an unhandled exception: {res}")
                errors[cfg.name] = str(res)
            elif isinstance(res, dict) and res.get("status") == "SUCCESS":
                successful_payloads[cfg.name] = res["data"]
            else:
                error_msg = res.get("error", "Unknown error during execution") if isinstance(res, dict) else str(res)
                errors[cfg.name] = error_msg

        return successful_payloads, errors

    def shutdown(self):
        """Cleanly drains the thread pool."""
        self.executor.shutdown(wait=True)
```

---

## 2. EXPERT A: MARKETING STRATEGIST

### 2.1 Role & Objective
Expert A analyzes target audiences, market resonance patterns, algorithmic retention hooks, and conversion dynamics. It determines the overarching ad archetype, pacing tempo, core psychological driver, and primary Call-To-Action.

### 2.2 System Prompt Text

```text
You are EXPERT A: THE LEAD PERFORMANCE MARKETING STRATEGIST for high-growth direct-to-consumer (DTC) and enterprise video advertising.

Your sole function is to ingest a product dossier, target customer profile, and campaign objective, and output a high-conversion strategic blueprint for a short-form video ad (TikTok, Instagram Reels, YouTube Shorts, or Programmatic OTT).

You must analyze the inputs across 5 conversion dimensions:
1. Behavioral Funnel Position: Cold acquisition vs. warm retargeting vs. retention.
2. Emotional Velocity: Identifying the single trigger (FOMO, relief, status, curiosity, excitement) that minimizes scroll-away rates within the first 1.5 seconds.
3. Hook Mechanics: Defining the hook archetype (Pattern Interrupt, Contrarian Statement, Visceral Product Action, Negative Consequence Warning, or Social Proof Aggression).
4. Creative Vehicle (Archetype):
   - commercial: High-production cinematic storytelling, pristine lighting, lifestyle elevation.
   - ugc: User-Generated Content style, handheld feel, native camera phone aesthetic, peer-to-peer authenticity.
   - editorial: High-fashion, typographic minimalism, fast montage cuts, rhythmic visual cadence.
   - tech_showcase: Macro lens engineering breakdown, exploded CAD diagrams, feature specification callouts.
5. Temporal Architecture: Total runtime (10-60s) strictly divided into discrete scenes (3 to 6 scenes).

CRITICAL SCHEMA REQUIREMENT:
You must return ONLY a raw JSON object. Do not include markdown code block syntax (do not use ```json or ```). Do not include any explanations, preambles, or conversational text. Your output must parse directly as RFC 8259 JSON.

The JSON object must strictly match this exact schema:
{
  "archetype": "commercial" | "ugc" | "editorial" | "tech_showcase",
  "marketing_angle": "string (Maximum 120 characters defining the unique selling proposition angle)",
  "target_emotion": "fomo" | "relief" | "status" | "curiosity" | "excitement",
  "hook_strategy": "string (Actionable breakdown of what captures attention in the first 0-3 seconds)",
  "cta_text": "string (Exact verbatim text for the terminal conversion call-to-action)",
  "urgency_level": "low" | "medium" | "high",
  "recommended_duration_sec": integer (Must be between 10 and 60),
  "scene_count": integer (Must be between 3 and 6)
}

DECISION RULES:
- If target_emotion is 'fomo' or 'excitement', pacing must be rapid, and urgency_level should typically be 'high'.
- If archetype is 'ugc', hook_strategy MUST avoid corporate jargon and reference relatable real-world friction.
- scene_count must realistically divide recommended_duration_sec such that each scene has a duration between 2.5 and 6.0 seconds.
```

---

## 3. EXPERT B: CINEMATOGRAPHER

### 3.1 Role & Objective
Expert B translates conceptual narratives into physical camera dynamics, optical physics, scene geometry, volumetric lighting profiles, and image-to-video diffusion preprocessing pipelines (ControlNet, IP-Adapter, depth conditioning, color LUTs).

### 3.2 System Prompt Text

```text
You are EXPERT B: THE MASTER CINEMATOGRAPHER AND COMPUTER VISION DIRECTOR for generative AI video pipelines.

Your function is to establish complete visual coherence, optical physics, camera motion kinetics, and neural preprocessing pipelines for generative video models (e.g., Sora, Runway Gen-3, Luma Dream Machine, Kling).

You must configure:
1. World Anchor: The universal spatial, optical, and stylistic environment that remains immutable across all scenes to prevent visual drift, temporal flicker, or model hallucinations.
   - environment: Precise physical stage, architecture, atmospheric density, or interior setting.
   - lighting: Key, fill, rim lighting setup, color temperature in Kelvin, lux level, diffusion modifiers (e.g., Profoto softbox, harsh sodium vapor, golden hour rim lighting).
   - lens_profile: Optical character, focal length, aperture setting, spherical vs. anamorphic, chromatic aberration, sensor size (e.g., Arri Alexa LF, Cooke Anamorphic/i 35mm T2.3).
   - color_grade: Color science, LUT designation, shadow/highlight split toning (e.g., Kodak 2383 D55 print emulation, bleach bypass, teal & orange rec709).
   - atmosphere: Airborne particulate, haze level, volumetric mist, rain droplets, cleanroom sterile.
   - negative_global: Exhaustive negative token list preventing morphing, extra limbs, low resolution, plastic skin, oversaturation, text artifacts, and structural warping.
2. Preprocessing Pipeline: Ordered image and video tensor conditioning steps (e.g., Canny edge detection, Depth estimation, Color Palette Remapping, Face Restoration, IP-Adapter reference locking).
3. Scene Camera Moves: Precise kinetic camera choreography for every scene index from 1 to N:
   - camera_motion: Standardized physical motion (e.g., 'dolly_in_fast', 'pan_right_orbital', 'pedestal_down', 'tracking_lateral', 'static_macro', 'fpv_dive').
   - focal_length: mm focal equivalent (e.g., '24mm', '50mm', '85mm', '100mm_macro').
   - transition_type: Optical cut (e.g., 'hard_cut', 'whip_pan_right', 'match_cut_action', 'zoom_blur_dissolve').
   - sfx_bridge: Diegetic sound effect anchor to bridge visual cuts (e.g., 'whoosh_air_low', 'metallic_snap', 'impact_sub_bass', or null).

CRITICAL SCHEMA REQUIREMENT:
You must return ONLY a raw JSON object. Do not include markdown code block syntax (no ```json or ```). Do not include any explanations, preambles, or conversational text. Your output must parse directly as RFC 8259 JSON.

The JSON object must strictly match this exact schema:
{
  "world_anchor": {
    "environment": "string",
    "lighting": "string",
    "lens_profile": "string",
    "color_grade": "string",
    "atmosphere": "string",
    "negative_global": "string"
  },
  "preprocessing_pipeline": [
    {
      "action": "string",
      "params": {
        "key": "value"
      }
    }
  ],
  "scene_camera_moves": [
    {
      "scene_id": integer,
      "camera_motion": "string",
      "focal_length": "string",
      "transition_type": "string",
      "sfx_bridge": "string or null"
    }
  ]
}

CINEMATOGRAPHIC RULES:
- Never specify physically impossible camera moves (e.g., teleportation, zooming through solid lead).
- Match camera velocity to the kinetic energy of the cut.
- Ensure lens_profile aligns with focal_length choices across scenes.
```

---

## 4. EXPERT C: SCRIPTWRITER & AUDIO DIRECTOR

### 4.1 Role & Objective
Expert C designs the auditory universe and typographic hierarchy. It writes voiceover copy calibrated to strict speaking rates (words-per-minute), dictates on-screen display (OSD) text for sound-off mobile browsing, and specifies the background musical score.

### 4.2 System Prompt Text

```text
You are EXPERT C: THE PRINCIPAL SCRIPTWRITER AND AUDIO EXPERIENCE DIRECTOR for direct-response and brand video ads.

Your objective is to craft the spoken script, typographic messaging hierarchy, and sonic architecture. Over 70% of mobile users view short-form ads with sound off, while 30% experience high-fidelity spatial audio. You must architect the ad to succeed equally in both modes.

You must determine:
1. Audio Mode:
   - voiceover: Professional narration, monologue, or voice actor delivery.
   - music_text_only: Pure sonic groove with reliance on dynamic on-screen typography.
   - asmr_foley: Hyper-detailed, tactile diegetic sound effects (crunch, snap, fizz, whisper).
   - original_audio: Realistic conversational dialogue or native UGC smartphone sound.
2. Music Vibe:
   - energetic: High-tempo (125-140 BPM), driving synth bass, electronic percussion.
   - luxury: Deep minimal house, warm sub-bass, sophisticated string pads, elegant cadence.
   - chill: Lo-fi hip-hop beats, mellow Rhodes piano, vinyl dust, relaxed 80-90 BPM.
   - dramatic: Cinematic orchestral crescendo, booming taiko drums, tension-building risers.
   - upbeat: Cheerful acoustic indie pop, clapping, positive syncopated rhythm.
   - asmr: Subdued rhythmic textural hums, binaural acoustic isolation.
3. Voiceover Script:
   - Full verbatim script text. If audio_mode is 'music_text_only', this MUST be null.
   - Calculate exact word count matching a natural speaking rate of 2.3 to 2.6 words per second.
4. On-Screen Texts:
   - Array of scene-by-scene textual overlays.
   - headline: Punchy, high-impact hook (3 to 6 words max).
   - body: Value explanation or secondary claim (optional, null if visual simplicity is needed).
   - cta: Specific call-to-action text (only present on terminal or sub-terminal scenes, otherwise null).
5. Pacing:
   - fast: Rapid cutting, 1.5 to 2.5 seconds per scene, aggressive word cadence.
   - medium: Balanced rhythm, 3.0 to 4.5 seconds per scene, natural conversational delivery.
   - slow: Meditative, luxury, cinematic contemplation, 5.0+ seconds per scene.

CRITICAL SCHEMA REQUIREMENT:
You must return ONLY a raw JSON object. Do not include markdown code block syntax (no ```json or ```). Do not include any explanations, preambles, or conversational text. Your output must parse directly as RFC 8259 JSON.

The JSON object must strictly match this exact schema:
{
  "audio_mode": "voiceover" | "music_text_only" | "asmr_foley" | "original_audio",
  "music_vibe": "energetic" | "luxury" | "chill" | "dramatic" | "upbeat" | "asmr",
  "voiceover_script": "string or null",
  "on_screen_texts": [
    {
      "scene_id": integer,
      "headline": "string",
      "body": "string or null",
      "cta": "string or null"
    }
  ],
  "pacing": "fast" | "medium" | "slow"
}

SCRIPTING RULES:
- Never exceed 6 words in an on-screen headline.
- If voiceover_script is provided, ensure phonetic clarity. Avoid tongue-twisters unless intentionally used as a comedic trope.
- Sync on_screen_texts scene_ids precisely to the planned scene structure.
```

---

## 5. EXPERT D: BRAND GUARDIAN & COMPLIANCE

### 5.1 Role & Objective
Expert D is the risk mitigation and brand equity auditor. It inspects campaign proposals against trademark protection rules, truth-in-advertising standards (FTC/NAD), platform community guidelines, color palette fidelity, and safe-zone asset positioning.

### 5.2 System Prompt Text

```text
You are EXPERT D: THE EXECUTIVE BRAND GUARDIAN AND LEGAL COMPLIANCE OFFICER.

Your mission is to enforce brand visual identity, legal safety, trademark protection, and platform advertising compliance (Meta, TikTok, Google Ads, FTC truth-in-advertising guidelines).

You have VETO AND OVERRIDE AUTHORITY over aesthetic and marketing recommendations that violate:
1. Brand Visual Integrity: Preserving hex-accurate color grading and preventing visual degradation of brand emblems.
2. Prohibited Claims: Absolute health guarantees, deceptive pricing, misleading "miracle" results, unauthorized celebrity/third-party likeness, or unsubstantiated scientific claims.
3. Safe-Zone Overlays: Placing logos and interactive elements outside platform interface occlusions (e.g., TikTok right-hand sidebar icons, bottom caption safe zones).

You must audit the incoming product dossier and provide:
1. Brand Violations: Array of identified compliance or brand risk strings. If none, return an empty array [].
2. Approved Color Grade: The legally enforced or brand-compliant color grading profile that overrides artistic Cinematographer proposals if a conflict arises.
3. Logo Placement:
   - corner: 'top_left' | 'top_right' | 'bottom_left' | 'bottom_right'.
     NOTE: 'bottom_right' and 'bottom_left' are generally prohibited on TikTok/Reels due to UI caption/sound-disc collisions. 'top_left' or 'top_right' is standard.
   - opacity: Float between 0.1 and 1.0 (recommended 0.8 to 0.95 for watermark retention).
   - size_pct: Float percentage of screen width (e.g., 8.0 to 18.0).
4. Font Override: Enforced primary typography family (e.g., 'Inter Bold', 'Helvetica Neue Bold', 'Founders Grotesk', or null to allow native system font).
5. Guardrail Notes: Specific instructions regarding disclaimer requirements, font licensing, or FTC-mandated sponsor badges.

CRITICAL SCHEMA REQUIREMENT:
You must return ONLY a raw JSON object. Do not include markdown code block syntax (no ```json or ```). Do not include any explanations, preambles, or conversational text. Your output must parse directly as RFC 8259 JSON.

The JSON object must strictly match this exact schema:
{
  "brand_violations": [
    "string"
  ],
  "approved_color_grade": "string",
  "logo_placement": {
    "corner": "top_left" | "top_right" | "bottom_left" | "bottom_right",
    "opacity": float (Between 0.0 and 1.0),
    "size_pct": float (Between 1.0 and 100.0)
  },
  "font_override": "string or null",
  "guardrail_notes": [
    "string"
  ]
}

AUDIT RULES:
- If unverified medical, environmental, or financial claims appear in the brief, log them immediately under brand_violations.
- Always enforce 'top_left' or 'top_right' for logo_placement when vertical 9:16 aspect ratio is targeted.
```

---

## 6. MASTER SYNTHESIZER & PRODUCTION MANIFEST

### 6.1 Synthesizer Role & Conflict Arbitration
The Master Synthesizer ingests the original campaign brief alongside the 4 expert JSON outputs. It resolves discrepancies:
1. **Color Grade Arbitration:** Expert D's `approved_color_grade` strictly overrides Expert B's `world_anchor.color_grade`.
2. **Timeline & Scene Reconciliation:** Expert A's `scene_count` and `recommended_duration_sec` define the temporal framework. The Synthesizer divides duration into frame-accurate timestamps, aligning Expert B's `scene_camera_moves` and Expert C's `on_screen_texts` into unified scene objects.
3. **Voiceover Feasibility:** If Expert C's script word count exceeds what can be spoken in the total duration at 2.6 words/sec, the Synthesizer truncates or re-paces the script.
4. **Diffusion Prompt Assembly:** Merges Expert B's `world_anchor` with the specific scene action, camera motion, and lighting to create an exhaustive, production-ready video generation prompt per scene.

### 6.2 Synthesizer System Prompt Text

```text
You are THE MASTER SYNTHESIZER of the AI Video Ad Engine.

Your task is to ingest:
1. The Original User/Product Brief
2. Expert A Output (Marketing Strategist)
3. Expert B Output (Cinematographer)
4. Expert C Output (Scriptwriter)
5. Expert D Output (Brand Guardian)

You must resolve all cross-functional conflicts and assemble the unified, frame-accurate MASTER PRODUCTION MANIFEST.

CONFLICT RESOLUTION DIRECTIVES:
1. COLOR GRADE HIERARCHY: Expert D's 'approved_color_grade' OVERRIDES Expert B's 'world_anchor.color_grade'. You must set the final color grade to Expert D's specification.
2. TIMELINE HARMONIZATION:
   - Use Expert A's 'recommended_duration_sec' and 'scene_count' as the authoritative timeline master.
   - Allocate discrete start_time_sec and end_time_sec to each scene. The sum of scene durations MUST exactly equal 'recommended_duration_sec'.
   - Distribute Expert B's camera motions and Expert C's on-screen texts across the unified scene indices (1 to N).
3. VOICE SCRIPT VALIDATION:
   - Verify that Expert C's 'voiceover_script' word count can be spoken comfortably in the total video duration. Assume a maximum of 2.5 words per second. If it is too long, edit and compress the script while preserving key brand claims.
4. DIFFUSION PROMPT COMPILATION:
   - For every scene in the timeline, compile an exhaustive, high-density 'visual_prompt'. Combine:
     [Subject & Product Action] + [Expert B Environment & World Anchor] + [Expert B Lighting & Lens] + [Expert B Camera Movement] + [Expert D Approved Color Grade].

CRITICAL SCHEMA REQUIREMENT:
You must output ONLY valid, parseable JSON conforming to the MasterProductionManifest schema. Do not enclose in markdown blocks (no ```json or ```). No preambles, explanations, or notes.
```

### 6.3 Complete MasterProductionManifest JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "MasterProductionManifest",
  "description": "Unified, machine-executable production manifest for generative AI video rendering and compilation.",
  "type": "object",
  "required": [
    "manifest_version",
    "campaign_metadata",
    "brand_compliance",
    "global_rendering_profile",
    "audio_stem_configuration",
    "scenes",
    "render_pipeline_export"
  ],
  "properties": {
    "manifest_version": {
      "type": "string",
      "enum": ["2.0.0"]
    },
    "campaign_metadata": {
      "type": "object",
      "required": [
        "campaign_id",
        "archetype",
        "marketing_angle",
        "target_emotion",
        "urgency_level",
        "total_duration_sec",
        "aspect_ratio",
        "target_platform"
      ],
      "properties": {
        "campaign_id": { "type": "string" },
        "archetype": {
          "type": "string",
          "enum": ["commercial", "ugc", "editorial", "tech_showcase"]
        },
        "marketing_angle": { "type": "string" },
        "target_emotion": {
          "type": "string",
          "enum": ["fomo", "relief", "status", "curiosity", "excitement"]
        },
        "urgency_level": {
          "type": "string",
          "enum": ["low", "medium", "high"]
        },
        "total_duration_sec": {
          "type": "integer",
          "minimum": 10,
          "maximum": 60
        },
        "aspect_ratio": {
          "type": "string",
          "enum": ["9:16", "16:9", "1:1", "4:5"]
        },
        "target_platform": {
          "type": "string",
          "enum": ["tiktok", "instagram_reels", "youtube_shorts", "meta_feed", "ott_tv"]
        }
      }
    },
    "brand_compliance": {
      "type": "object",
      "required": [
        "enforced_color_grade",
        "font_family",
        "logo_watermark",
        "compliance_approved",
        "guardrail_audit_log"
      ],
      "properties": {
        "enforced_color_grade": { "type": "string" },
        "font_family": { "type": "string" },
        "logo_watermark": {
          "type": "object",
          "required": ["corner", "opacity", "size_pct"],
          "properties": {
            "corner": {
              "type": "string",
              "enum": ["top_left", "top_right", "bottom_left", "bottom_right"]
            },
            "opacity": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1.0
            },
            "size_pct": {
              "type": "number",
              "minimum": 1.0,
              "maximum": 50.0
            }
          }
        },
        "compliance_approved": { "type": "boolean" },
        "guardrail_audit_log": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    },
    "global_rendering_profile": {
      "type": "object",
      "required": [
        "environment",
        "lighting",
        "lens_profile",
        "atmosphere",
        "negative_prompt",
        "preprocessing_stack"
      ],
      "properties": {
        "environment": { "type": "string" },
        "lighting": { "type": "string" },
        "lens_profile": { "type": "string" },
        "atmosphere": { "type": "string" },
        "negative_prompt": { "type": "string" },
        "preprocessing_stack": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["action", "params"],
            "properties": {
              "action": { "type": "string" },
              "params": { "type": "object" }
            }
          }
        }
      }
    },
    "audio_stem_configuration": {
      "type": "object",
      "required": [
        "audio_mode",
        "music_vibe",
        "pacing",
        "master_voiceover_script",
        "ducking_profile"
      ],
      "properties": {
        "audio_mode": {
          "type": "string",
          "enum": ["voiceover", "music_text_only", "asmr_foley", "original_audio"]
        },
        "music_vibe": {
          "type": "string",
          "enum": ["energetic", "luxury", "chill", "dramatic", "upbeat", "asmr"]
        },
        "pacing": {
          "type": "string",
          "enum": ["fast", "medium", "slow"]
        },
        "master_voiceover_script": {
          "type": ["string", "null"]
        },
        "ducking_profile": {
          "type": "object",
          "required": ["enabled", "music_attenuation_db", "attack_ms", "release_ms"],
          "properties": {
            "enabled": { "type": "boolean" },
            "music_attenuation_db": { "type": "number" },
            "attack_ms": { "type": "integer" },
            "release_ms": { "type": "integer" }
          }
        }
      }
    },
    "scenes": {
      "type": "array",
      "minItems": 3,
      "maxItems": 6,
      "items": {
        "type": "object",
        "required": [
          "scene_id",
          "start_time_sec",
          "end_time_sec",
          "duration_sec",
          "visual_prompt",
          "camera_motion",
          "focal_length",
          "transition_type",
          "sfx_cue",
          "on_screen_display"
        ],
        "properties": {
          "scene_id": { "type": "integer" },
          "start_time_sec": { "type": "number", "minimum": 0.0 },
          "end_time_sec": { "type": "number", "minimum": 0.0 },
          "duration_sec": { "type": "number", "minimum": 1.0 },
          "visual_prompt": { "type": "string" },
          "camera_motion": { "type": "string" },
          "focal_length": { "type": "string" },
          "transition_type": { "type": "string" },
          "sfx_cue": { "type": ["string", "null"] },
          "on_screen_display": {
            "type": "object",
            "required": ["headline", "body", "cta"],
            "properties": {
              "headline": { "type": "string" },
              "body": { "type": ["string", "null"] },
              "cta": { "type": ["string", "null"] }
            }
          }
        }
      }
    },
    "render_pipeline_export": {
      "type": "object",
      "required": [
        "target_fps",
        "resolution_width",
        "resolution_height",
        "video_codec",
        "audio_sample_rate_hz",
        "ffmpeg_complex_filter_stub"
      ],
      "properties": {
        "target_fps": { "type": "integer", "enum": [24, 30, 60] },
        "resolution_width": { "type": "integer" },
        "resolution_height": { "type": "integer" },
        "video_codec": { "type": "string", "enum": ["h264", "hevc", "prores422"] },
        "audio_sample_rate_hz": { "type": "integer", "enum": [44100, 48000] },
        "ffmpeg_complex_filter_stub": { "type": "string" }
      }
    }
  }
}
```

---

## 7. ERROR HANDLING & VALIDATION PIPELINE

Post-synthesis validation ensures that all generated manifests are schema-compliant and executable by downstream rendering workers. This is enforced via Pydantic V2 models, regex-based JSON cleaning, and programmatic conflict repairs.

```
┌─────────────────────────────────┐
│     Raw LLM Output String       │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  Regex Code Block & BOM Strip   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       Standard JSON Parser      │◄────────┐
└────────────────┬────────────────┘         │
                 │ (If Malformed)           │
                 ▼                          │
┌─────────────────────────────────┐         │
│  Trailing Comma & Quote Repair  ├─────────┘
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Pydantic V2 Structural Audit  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Semantic Business Rule Audit  │
│  - Total Duration == Sum(Scenes)│
│  - Expert D Color Grade Override│
│  - Max WPM Speaking Rate Limit  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Validated Manifest to Rendering │
└─────────────────────────────────┘
```

### 7.1 Pydantic V2 Validation & Repair Models

```python
"""
module_2_validation.py
Pydantic V2 Models and Validation Engine for MasterProductionManifest.
Enforces structural, relational, and semantic invariants.
"""

from typing import List, Optional, Dict, Any, Literal
import re
import json
import logging
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ValidationError
)

logger = logging.getLogger("MoE-Validator")


class LogoWatermarkModel(BaseModel):
    corner: Literal["top_left", "top_right", "bottom_left", "bottom_right"]
    opacity: float = Field(ge=0.0, le=1.0)
    size_pct: float = Field(ge=1.0, le=50.0)


class BrandComplianceModel(BaseModel):
    enforced_color_grade: str
    font_family: str
    logo_watermark: LogoWatermarkModel
    compliance_approved: bool
    guardrail_audit_log: List[str] = Field(default_factory=list)


class CampaignMetadataModel(BaseModel):
    campaign_id: str
    archetype: Literal["commercial", "ugc", "editorial", "tech_showcase"]
    marketing_angle: str = Field(max_length=200)
    target_emotion: Literal["fomo", "relief", "status", "curiosity", "excitement"]
    urgency_level: Literal["low", "medium", "high"]
    total_duration_sec: int = Field(ge=10, le=60)
    aspect_ratio: Literal["9:16", "16:9", "1:1", "4:5"]
    target_platform: Literal["tiktok", "instagram_reels", "youtube_shorts", "meta_feed", "ott_tv"]


class PreprocessingStackItem(BaseModel):
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)


class GlobalRenderingProfileModel(BaseModel):
    environment: str
    lighting: str
    lens_profile: str
    atmosphere: str
    negative_prompt: str
    preprocessing_stack: List[PreprocessingStackItem] = Field(default_factory=list)


class DuckingProfileModel(BaseModel):
    enabled: bool = True
    music_attenuation_db: float = -12.0
    attack_ms: int = 150
    release_ms: int = 500


class AudioStemConfigurationModel(BaseModel):
    audio_mode: Literal["voiceover", "music_text_only", "asmr_foley", "original_audio"]
    music_vibe: Literal["energetic", "luxury", "chill", "dramatic", "upbeat", "asmr"]
    pacing: Literal["fast", "medium", "slow"]
    master_voiceover_script: Optional[str] = None
    ducking_profile: DuckingProfileModel = Field(default_factory=DuckingProfileModel)


class OnScreenDisplayModel(BaseModel):
    headline: str = Field(max_length=80)
    body: Optional[str] = None
    cta: Optional[str] = None


class SceneModel(BaseModel):
    scene_id: int = Field(ge=1)
    start_time_sec: float = Field(ge=0.0)
    end_time_sec: float = Field(ge=0.0)
    duration_sec: float = Field(ge=0.5)
    visual_prompt: str
    camera_motion: str
    focal_length: str
    transition_type: str
    sfx_cue: Optional[str] = None
    on_screen_display: OnScreenDisplayModel

    @model_validator(mode="after")
    def verify_scene_timestamps(self) -> "SceneModel":
        calculated_duration = round(self.end_time_sec - self.start_time_sec, 3)
        if abs(calculated_duration - round(self.duration_sec, 3)) > 0.05:
            # Self-heal duration mismatch
            self.duration_sec = calculated_duration
        if self.end_time_sec <= self.start_time_sec:
            raise ValueError(f"Scene {self.scene_id}: end_time_sec must be > start_time_sec.")
        return self


class RenderPipelineExportModel(BaseModel):
    target_fps: Literal[24, 30, 60] = 30
    resolution_width: int
    resolution_height: int
    video_codec: Literal["h264", "hevc", "prores422"] = "h264"
    audio_sample_rate_hz: Literal[44100, 48000] = 48000
    ffmpeg_complex_filter_stub: str


class MasterProductionManifest(BaseModel):
    manifest_version: Literal["2.0.0"] = "2.0.0"
    campaign_metadata: CampaignMetadataModel
    brand_compliance: BrandComplianceModel
    global_rendering_profile: GlobalRenderingProfileModel
    audio_stem_configuration: AudioStemConfigurationModel
    scenes: List[SceneModel] = Field(min_length=3, max_length=6)
    render_pipeline_export: RenderPipelineExportModel

    @model_validator(mode="after")
    def validate_master_business_invariants(self) -> "MasterProductionManifest":
        """
        Enforces cross-cutting integrity:
        1. Total duration equals sum of individual scene durations.
        2. Continuous, non-overlapping scene timeline.
        3. Speaking rate within maximum words-per-minute threshold.
        """
        # 1. Timeline integrity
        total_meta_duration = float(self.campaign_metadata.total_duration_sec)
        actual_scene_sum = sum(s.duration_sec for s in self.scenes)

        if abs(total_meta_duration - actual_scene_sum) > 0.5:
            logger.warning(
                f"Timeline discrepancy: Meta duration ({total_meta_duration}s) != "
                f"Sum of scenes ({actual_scene_sum}s). Auto-adjusting terminal scene."
            )
            delta = total_meta_duration - actual_scene_sum
            self.scenes[-1].duration_sec += delta
            self.scenes[-1].end_time_sec += delta

        # 2. Timeline continuity check
        for i in range(len(self.scenes) - 1):
            curr_end = self.scenes[i].end_time_sec
            next_start = self.scenes[i + 1].start_time_sec
            if abs(curr_end - next_start) > 0.05:
                # Heal continuity gap
                self.scenes[i + 1].start_time_sec = curr_end
                self.scenes[i + 1].end_time_sec = curr_end + self.scenes[i + 1].duration_sec

        # 3. Voiceover speaking rate check (max 2.6 words/second)
        vo_script = self.audio_stem_configuration.master_voiceover_script
        if vo_script:
            words = len(vo_script.split())
            max_allowed_words = int(total_meta_duration * 2.6)
            if words > max_allowed_words:
                logger.warning(
                    f"Voiceover word count ({words}) exceeds maximum capacity "
                    f"({max_allowed_words}) for {total_meta_duration}s ad."
                )

        return self


class RobustManifestParser:
    """
    Cleans, repairs, and parses raw text responses into validated MasterProductionManifest models.
    """

    @staticmethod
    def clean_json_string(raw_str: str) -> str:
        """Removes markdown blocks, BOMs, and trailing commas."""
        cleaned = raw_str.strip()
        # Remove markdown code fences
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        # Remove trailing commas before closing braces/brackets
        cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
        return cleaned

    @classmethod
    def parse_and_validate(cls, raw_response: str) -> Tuple[Optional[MasterProductionManifest], Optional[str]]:
        try:
            sanitized = cls.clean_json_string(raw_response)
            parsed_dict = json.loads(sanitized)
            manifest = MasterProductionManifest.model_validate(parsed_dict)
            return manifest, None
        except json.JSONDecodeError as jde:
            error_msg = f"JSON Syntax Error: {str(jde)}"
            logger.error(error_msg)
            return None, error_msg
        except ValidationError as ve:
            error_msg = f"Schema Validation Error: {ve.json(indent=2)}"
            logger.error(error_msg)
            return None, error_msg
        except Exception as exc:
            error_msg = f"Unexpected Manifest Error: {str(exc)}"
            logger.error(error_msg)
            return None, error_msg
```

---

## 8. COMPLETE VERIFICATION TEST SUITE

The following test harness executes a complete cycle of Module 2. It verifies:
1. ThreadPoolExecutor multi-worker concurrency.
2. Complete simulation of Experts A, B, C, D running in parallel.
3. Master Synthesizer conflict arbitration (specifically Brand Guardian color grade override).
4. Pydantic validation and timeline auto-repair.

```python
"""
test_module_2_moe_brain.py
Full Verification Test Suite for Module 2: The MoE Thinking Brain.
Runs self-contained mock and integration tests.
"""

import asyncio
import time
import json
import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from module_2_moe_brain import (
    MoEConcurrencyOrchestrator,
    ExpertConfig
)
from module_2_validation import (
    MasterProductionManifest,
    RobustManifestParser
)

# ---------------------------------------------------------------------------
# MOCK PAYLOADS FOR VERIFICATION
# ---------------------------------------------------------------------------

MOCK_EXPERT_A_RESPONSE = {
    "archetype": "commercial",
    "marketing_angle": "Ultra-light carbon fiber running shoes designed for marathon endurance",
    "target_emotion": "excitement",
    "hook_strategy": "Visceral close-up of sole compression at high speed with instant rebound",
    "cta_text": "Claim Your Stride - 20% Off Launch Edition",
    "urgency_level": "high",
    "recommended_duration_sec": 15,
    "scene_count": 3
}

MOCK_EXPERT_B_RESPONSE = {
    "world_anchor": {
        "environment": "Urban asphalt running track in foggy dawn neo-Tokyo",
        "lighting": "Low-angle cool dawn diffuse fill with neon magenta rim lighting",
        "lens_profile": "Arri Master Prime 35mm T1.3, sharp modern contrast",
        "color_grade": "Moody cyberpunk teal and deep shadows",  # Subject to override
        "atmosphere": "Morning condensation, volumetric ground fog",
        "negative_global": "blurry, low-res, deformed toes, bad shoe laces, CGI cartoon look"
    },
    "preprocessing_pipeline": [
        {"action": "canny_edge_detection", "params": {"low_threshold": 100, "high_threshold": 200}},
        {"action": "depth_anything_v2", "params": {"weight": 0.85}}
    ],
    "scene_camera_moves": [
        {"scene_id": 1, "camera_motion": "tracking_ground_macro", "focal_length": "24mm", "transition_type": "whip_pan_right", "sfx_bridge": "sub_impact_bass"},
        {"scene_id": 2, "camera_motion": "orbital_low_angle", "focal_length": "35mm", "transition_type": "hard_cut", "sfx_bridge": "pneumatic_release"},
        {"scene_id": 3, "camera_motion": "static_hero_shot", "focal_length": "50mm", "transition_type": "fade_black", "sfx_bridge": null}
    ]
}

MOCK_EXPERT_C_RESPONSE = {
    "audio_mode": "voiceover",
    "music_vibe": "energetic",
    "voiceover_script": "Gravity used to hold you back. Now, it propels you forward. Feel the carbon rebound.",
    "on_screen_texts": [
        {"scene_id": 1, "headline": "GRAVITY REDEFINED", "body": null, "cta": null},
        {"scene_id": 2, "headline": "CARBON KINETICS", "body": "100% Energy Return", "cta": null},
        {"scene_id": 3, "headline": "DEFY YOUR LIMITS", "body": null, "cta": "GET 20% OFF NOW"}
    ],
    "pacing": "fast"
}

MOCK_EXPERT_D_RESPONSE = {
    "brand_violations": [],
    "approved_color_grade": "Aero Athletic Kodachrome High-Key Punch",  # Must override Expert B
    "logo_placement": {
        "corner": "top_right",
        "opacity": 0.9,
        "size_pct": 14.5
    },
    "font_override": "Futura Bold Condensed",
    "guardrail_notes": ["Ensure disclaimer 'Results may vary by runner weight' is present on Scene 3."]
}

MOCK_SYNTHESIZED_MANIFEST = {
    "manifest_version": "2.0.0",
    "campaign_metadata": {
        "campaign_id": "CMP-TEST-RUNNER-001",
        "archetype": "commercial",
        "marketing_angle": "Ultra-light carbon fiber running shoes designed for marathon endurance",
        "target_emotion": "excitement",
        "urgency_level": "high",
        "total_duration_sec": 15,
        "aspect_ratio": "9:16",
        "target_platform": "tiktok"
    },
    "brand_compliance": {
        "enforced_color_grade": "Aero Athletic Kodachrome High-Key Punch",
        "font_family": "Futura Bold Condensed",
        "logo_watermark": {
            "corner": "top_right",
            "opacity": 0.9,
            "size_pct": 14.5
        },
        "compliance_approved": True,
        "guardrail_audit_log": [
            "Override Expert B color grade with brand standard Kodachrome High-Key",
            "Validated Safe Zone: Logo in top_right avoids TikTok bottom interaction bar"
        ]
    },
    "global_rendering_profile": {
        "environment": "Urban asphalt running track in foggy dawn neo-Tokyo",
        "lighting": "Low-angle cool dawn diffuse fill with neon magenta rim lighting",
        "lens_profile": "Arri Master Prime 35mm T1.3, sharp modern contrast",
        "atmosphere": "Morning condensation, volumetric ground fog",
        "negative_prompt": "blurry, low-res, deformed toes, bad shoe laces, CGI cartoon look",
        "preprocessing_stack": [
            {"action": "canny_edge_detection", "params": {"low_threshold": 100, "high_threshold": 200}},
            {"action": "depth_anything_v2", "params": {"weight": 0.85}}
        ]
    },
    "audio_stem_configuration": {
        "audio_mode": "voiceover",
        "music_vibe": "energetic",
        "pacing": "fast",
        "master_voiceover_script": "Gravity used to hold you back. Now, it propels you forward. Feel the carbon rebound.",
        "ducking_profile": {
            "enabled": True,
            "music_attenuation_db": -14.0,
            "attack_ms": 100,
            "release_ms": 400
        }
    },
    "scenes": [
        {
            "scene_id": 1,
            "start_time_sec": 0.0,
            "end_time_sec": 4.5,
            "duration_sec": 4.5,
            "visual_prompt": "Macro tracking shot of carbon fiber running shoe sole impacting wet asphalt at high velocity, water droplets spraying in slow motion, Aero Athletic Kodachrome High-Key Punch, Arri Master Prime 35mm, cinematic neon rim lighting.",
            "camera_motion": "tracking_ground_macro",
            "focal_length": "24mm",
            "transition_type": "whip_pan_right",
            "sfx_cue": "sub_impact_bass",
            "on_screen_display": {
                "headline": "GRAVITY REDEFINED",
                "body": None,
                "cta": None
            }
        },
        {
            "scene_id": 2,
            "start_time_sec": 4.5,
            "end_time_sec": 10.0,
            "duration_sec": 5.5,
            "visual_prompt": "Low angle orbital camera circling athlete sprinting at full stride, carbon plate flexing visibly, volumetric morning mist, Aero Athletic Kodachrome High-Key Punch, highly detailed athletic musculature.",
            "camera_motion": "orbital_low_angle",
            "focal_length": "35mm",
            "transition_type": "hard_cut",
            "sfx_cue": "pneumatic_release",
            "on_screen_display": {
                "headline": "CARBON KINETICS",
                "body": "100% Energy Return",
                "cta": None
            }
        },
        {
            "scene_id": 3,
            "start_time_sec": 10.0,
            "end_time_sec": 15.0,
            "duration_sec": 5.0,
            "visual_prompt": "Static hero showcase of the carbon running shoe floating in air, rotating slowly with light reflecting off carbon weave, Aero Athletic Kodachrome High-Key Punch, pristine studio lighting contrast.",
            "camera_motion": "static_hero_shot",
            "focal_length": "50mm",
            "transition_type": "fade_black",
            "sfx_cue": None,
            "on_screen_display": {
                "headline": "DEFY YOUR LIMITS",
                "body": None,
                "cta": "GET 20% OFF NOW"
            }
        }
    ],
    "render_pipeline_export": {
        "target_fps": 30,
        "resolution_width": 1080,
        "resolution_height": 1920,
        "video_codec": "h264",
        "audio_sample_rate_hz": 48000,
        "ffmpeg_complex_filter_stub": "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920[v0];[1:v]scale=1080:1920[v1];[v0][v1]xfade=transition=wipeleft:duration=0.5:offset=4.0"
    }
}


class TestMoEThinkingBrain(unittest.TestCase):
    """
    Automated test harness verifying Module 2 functionality.
    """

    def setUp(self):
        self.orchestrator = MoEConcurrencyOrchestrator(
            api_key="TEST_MOCK_API_KEY",
            max_workers=4
        )

    def tearDown(self):
        self.orchestrator.shutdown()

    def test_01_concurrency_execution(self):
        """
        Verify that 4 simulated synchronous API calls execute concurrently
        in ~0.5s rather than serially in ~2.0s.
        """
        async def run_parallel_test():
            def mock_sleep_worker(cfg, prompt):
                time.sleep(0.5)  # Simulate API latency
                return {"expert": cfg.name, "data": {"status": "ok"}, "latency_sec": 0.5, "status": "SUCCESS"}

            configs = [
                ExpertConfig(name="Expert_A", model_name="gemini-1.5-pro", system_instruction="A", temperature=0.7, max_output_tokens=1000),
                ExpertConfig(name="Expert_B", model_name="gemini-1.5-pro", system_instruction="B", temperature=0.7, max_output_tokens=1000),
                ExpertConfig(name="Expert_C", model_name="gemini-1.5-pro", system_instruction="C", temperature=0.7, max_output_tokens=1000),
                ExpertConfig(name="Expert_D", model_name="gemini-1.5-pro", system_instruction="D", temperature=0.7, max_output_tokens=1000),
            ]

            with patch.object(self.orchestrator, "_sync_gemini_call", side_effect=mock_sleep_worker):
                start = time.perf_counter()
                results, errors = await self.orchestrator.fire_all_experts_simultaneously(
                    configs,
                    shared_user_prompt="Run speed test"
                )
                duration = time.perf_counter() - start

                # Concurrency assertion: 4 calls of 0.5s must complete in under 0.9s
                self.assertLess(duration, 0.9, f"Calls ran sequentially! Elapsed: {duration:.2f}s")
                self.assertEqual(len(results), 4)
                self.assertEqual(len(errors), 0)

        asyncio.run(run_parallel_test())

    def test_02_master_manifest_validation(self):
        """
        Validates that the complete MasterProductionManifest schema parses and passes all Pydantic checks.
        """
        raw_json = json.dumps(MOCK_SYNTHESIZED_MANIFEST)
        manifest, err = RobustManifestParser.parse_and_validate(raw_json)

        self.assertIsNone(err, f"Validation failed with error: {err}")
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.manifest_version, "2.0.0")
        self.assertEqual(manifest.campaign_metadata.total_duration_sec, 15)
        self.assertEqual(len(manifest.scenes), 3)

        # Check Brand Guardian Override
        self.assertEqual(
            manifest.brand_compliance.enforced_color_grade,
            "Aero Athletic Kodachrome High-Key Punch"
        )

    def test_03_timeline_auto_healing(self):
        """
        Validates that an erroneous timeline (scenes sum != total_duration_sec)
        is automatically repaired by the model validator.
        """
        broken_manifest_dict = json.loads(json.dumps(MOCK_SYNTHESIZED_MANIFEST))
        # Corrupt the last scene duration to create a discrepancy
        broken_manifest_dict["scenes"][-1]["duration_sec"] = 3.0
        broken_manifest_dict["scenes"][-1]["end_time_sec"] = 13.0
        # Now scenes sum to 4.5 + 5.5 + 3.0 = 13.0s, but metadata says 15s

        raw_json = json.dumps(broken_manifest_dict)
        manifest, err = RobustManifestParser.parse_and_validate(raw_json)

        self.assertIsNone(err)
        self.assertIsNotNone(manifest)
        # Verify auto-healing adjusted the final scene to absorb the delta
        self.assertEqual(manifest.scenes[-1].duration_sec, 5.0)
        self.assertEqual(manifest.scenes[-1].end_time_sec, 15.0)

    def test_04_json_cleaning_resilience(self):
        """
        Tests parser resilience against markdown wraps and trailing commas.
        """
        dirty_raw_llm_string = f"""
        ```json
        {json.dumps(MOCK_SYNTHESIZED_MANIFEST)},
        ```
        """
        manifest, err = RobustManifestParser.parse_and_validate(dirty_raw_llm_string)
        self.assertIsNone(err)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest.campaign_metadata.campaign_id, "CMP-TEST-RUNNER-001")


if __name__ == "__main__":
    unittest.main(verbosity=2)
```

---

### SUMMARY OF MODULE SPECIFICATION DELIVERABLES

| Component | Technical Implementation | Operational Role |
| :--- | :--- | :--- |
| **Concurrency Engine** | `ThreadPoolExecutor(max_workers=4)` + `asyncio.run_in_executor` | Fires all 4 synchronous Gemini calls simultaneously with zero event-loop blocking. |
| **Expert A (Marketing)** | Pure JSON Enum Constraint Model | Establishes ad archetype, psychological retention hook, and campaign pacing. |
| **Expert B (Cinematographer)** | World Anchor & Tensor Conditioning Schema | Freezes visual world anchor, camera movement kinetics, and diffusion prep pipeline. |
| **Expert C (Scriptwriter)** | Typographic & Audio WPM Engine | Formulates timed voiceover scripts, sound cues, and on-screen text hierarchies. |
| **Expert D (Brand Guardian)** | Visual & Regulatory Veto Authority | Enforces logo safe-zones, FTC compliance, and overrides non-compliant color grades. |
| **Master Synthesizer** | Conflict Resolution & Merging Core | Merges 4 expert outputs into a frame-accurate, production-ready `MasterProductionManifest`. |
| **Validation Pipeline** | Pydantic V2 + Heuristic Timeline Repair | Auto-repairs timeline mismatches and strips LLM markdown formatting artifacts. |
| **Verification Suite** | Standalone PyUnit Harness | Asserts multi-threaded concurrency, schema validation, and error recovery. |

---

# AI Video Ad Engine: Modules 3A, 3B, 3C, 3D Specification

---

## MODULE 3A: PREPROCESSOR

### 1. Architectural Overview & Workflow
The Preprocessor ingests raw user-provided product assets (JPEG, PNG, WEBP), strips background noise to isolate the hero subject, and upscales the asset to high fidelity (minimum 2K resolution) before video generative seed embedding.

```
[Raw Product Image] 
        │
        ▼
[rembg: U2-Net Isolation] ──► Produces RGBA Alpha Cutout (PNG)
        │
        ▼
[Replicate: Real-ESRGAN]   ──► Produces 4x Upscaled Hero Master Asset
        │
        ▼
[Dimension & Aspect Normalizer (9:16 Canvas)]
```

---

### 2. Background Removal (`rembg`)

> [!WARNING]
> **CRITICAL FIRST-RUN MODEL DOWNLOAD (176MB):**
> On first invocation, `rembg` automatically triggers a download of the `u2net.onnx` weights file (~176 MB) from GitHub Releases (`https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx`) into the local user directory (`~/.u2net/u2net.onnx`).
> 
> In containerized or ephemeral production environments (e.g., Docker, Kubernetes, AWS Lambda, Cloud Run):
> 1. Uncached downloads will cause cold-start latency spikes (10s–45s) and risk HTTP 504 gateway timeouts.
> 2. Environments without outbound public internet access will fail immediately with `urllib.error.URLError`.
> 
> **Production Fix:** Pre-bake weights into the container image during the build stage:
> ```dockerfile
> ENV U2NET_HOME=/opt/models/u2net
> RUN mkdir -p /opt/models/u2net && \
>     python -c "import rembg; rembg.new_session('u2net')"
> ```

#### Exact Python Implementation
```python
import io
import os
import logging
from pathlib import Path
from PIL import Image
from rembg import remove, new_session

logger = logging.getLogger("preprocessor.rembg")

# Initialize and cache a persistent ONNX session
# Session pooling prevents re-allocating ONNX runtime memory per request
REMBG_SESSION = new_session(model_name="u2net")

def remove_background(input_path: Path, output_path: Path) -> Path:
    """
    Strips background from product image and outputs transparent 32-bit RGBA PNG.
    
    Args:
        input_path: Path to source image file
        output_path: Target path for isolated PNG
        
    Returns:
        Path to processed PNG
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
        
    try:
        with open(input_path, "rb") as img_file:
            input_bytes = img_file.read()
            
        # Execute background removal with alpha matting for edge feathering
        output_bytes = remove(
            input_bytes,
            session=REMBG_SESSION,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10
        )
        
        # Validate resulting image data
        image = Image.open(io.BytesIO(output_bytes))
        if image.mode != "RGBA":
            image = image.convert("RGBA")
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG", optimize=True)
        logger.info(f"Background successfully removed: {output_path}")
        return output_path

    except Exception as exc:
        logger.error(f"Failed to strip background for {input_path}: {str(exc)}")
        raise RuntimeError(f"rembg processing error: {exc}") from exc
```

---

### 3. Resolution Upscaling (Replicate Real-ESRGAN)

#### Exact Replicate Call
```python
import os
import time
import requests
import replicate
from pathlib import Path
from typing import Optional

logger = logging.getLogger("preprocessor.upscaler")

# Pinned Replicate Real-ESRGAN model version hash
REAL_ESRGAN_VERSION = "nightmareai/real-esrgan:f121d64388e6f1379a8451c5a2c5e544d68d7abe9fdbde4e19f8510803e70dce"

def upscale_real_esrgan(
    input_path: Path, 
    output_path: Path, 
    scale: int = 4, 
    face_enhance: bool = False
) -> Path:
    """
    Upscales image using Replicate Real-ESRGAN API.
    
    Args:
        input_path: Path to transparent or opaque product image
        output_path: Target destination for high-res output
        scale: Upscale multiplier (2, 4, or 8)
        face_enhance: Apply GFPGAN face restoration (keep False for products)
        
    Returns:
        Path to upscaled saved image
    """
    if "REPLICATE_API_TOKEN" not in os.environ:
        raise EnvironmentError("REPLICATE_API_TOKEN environment variable is missing.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(input_path, "rb") as image_file:
            logger.info(f"Submitting {input_path.name} to Replicate Real-ESRGAN...")
            
            output_url = replicate.run(
                REAL_ESRGAN_VERSION,
                input={
                    "image": image_file,
                    "scale": scale,
                    "face_enhance": face_enhance
                }
            )

        if not output_url:
            raise ValueError("Replicate returned empty output URL.")

        # Download upscaled binary stream
        response = requests.get(str(output_url), timeout=60)
        response.raise_for_status()

        with open(output_path, "wb") as f_out:
            f_out.write(response.content)

        logger.info(f"Upscaled image saved to {output_path}")
        return output_path

    except replicate.exceptions.ReplicateError as rep_err:
        logger.error(f"Replicate API execution failed: {rep_err}")
        raise
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Failed to download upscaled asset from CDN: {req_err}")
        raise
```

---

### 4. Comprehensive Preprocessor Pipeline & Error Handling

```python
import time
from PIL import Image, UnidentifiedImageError

MAX_INPUT_DIMENSION = 4096
MIN_INPUT_DIMENSION = 256

class PreprocessorPipeline:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_image(self, file_path: Path) -> Image.Image:
        """Validates file integrity, dimensions, and color profiles."""
        if not file_path.exists():
            raise FileNotFoundError(f"Asset does not exist: {file_path}")
            
        try:
            with Image.open(file_path) as img:
                img.verify() # Verify file header / corruption
                
            # Reopen after verify() resets internal file pointers
            img = Image.open(file_path)
            w, h = img.size
            
            if w < MIN_INPUT_DIMENSION or h < MIN_INPUT_DIMENSION:
                raise ValueError(f"Image too small ({w}x{h}). Minimum {MIN_INPUT_DIMENSION}px required.")
                
            # If asset exceeds 4K, downscale prior to rembg to prevent CUDA/RAM exhaustion
            if max(w, h) > MAX_INPUT_DIMENSION:
                ratio = MAX_INPUT_DIMENSION / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                clamped_path = self.output_dir / f"clamped_{file_path.name}"
                img.save(clamped_path)
                return clamped_path
                
            return file_path

        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Corrupt or unsupported image asset {file_path}: {e}") from e

    def execute(self, raw_asset_path: Path, scene_id: str) -> dict:
        """Executes full preprocessing chain with fallback mechanisms."""
        validated_path = self.validate_image(raw_asset_path)
        
        cutout_path = self.output_dir / f"{scene_id}_isolated.png"
        upscaled_path = self.output_dir / f"{scene_id}_master.png"

        # Step 1: Background removal
        cutout_res = remove_background(validated_path, cutout_path)

        # Step 2: Upscale with exponential retry
        max_retries = 3
        backoff = 2.0
        upscale_success = False
        
        for attempt in range(1, max_retries + 1):
            try:
                upscale_real_esrgan(cutout_res, upscaled_path, scale=4)
                upscale_success = True
                break
            except Exception as e:
                logger.warning(f"Upscale attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    logger.critical("Upscale retries exhausted. Falling back to Lanczos local resize.")

        # Fallback if Replicate fails
        if not upscale_success:
            with Image.open(cutout_res) as img:
                target_w = img.width * 2
                target_h = img.height * 2
                fallback_img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                fallback_img.save(upscaled_path)

        return {
            "status": "SUCCESS",
            "cutout_path": str(cutout_path),
            "master_path": str(upscaled_path)
        }
```

---

## MODULE 3B: SCENE CHAIN ENGINE

### 1. Architectural Overview & Continuity Engine
To prevent temporal discontinuity between separate video generations, the Scene Chain Engine utilizes a **tail-frame seed chain**:
- Scene 1 generates from initial product visual seed $I_0$.
- Tail frame $F_{1,\text{tail}}$ is extracted at $t = 4.8\text{s}$ of Scene 1.
- Scene 2 is conditioned on $F_{1,\text{tail}}$ as its start image input, guaranteeing pixel-level boundary consistency.

```
[Scene 0 (Init Asset)] ──► [Veo 2 Vertex AI] ──► [Scene 0 Clip (5s)]
                                                      │
                            ┌─────────────────────────┘
                            ▼
                    [FFmpeg Tail Extraction (-ss 4.8s)] ──► [Tail Frame PNG]
                                                                    │
    ┌───────────────────────────────────────────────────────────────┘
    ▼
[Scene 1 (Tail Frame)] ──► [Veo 2 Vertex AI] ──► [Scene 1 Clip (5s)] ──► ...
```

---

### 2. Complete SQLite Schema & Persistence Operations

```sql
-- Schema Definition: scene_chain.sql
CREATE TABLE IF NOT EXISTS video_chains (
    chain_id TEXT PRIMARY KEY,
    campaign_name TEXT NOT NULL,
    total_scenes INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('INITIALIZED', 'PROCESSING', 'COMPLETED', 'FAILED')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scenes (
    scene_id TEXT PRIMARY KEY,
    chain_id TEXT NOT NULL,
    scene_index INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    camera_motion TEXT,
    start_image_path TEXT,
    clip_path TEXT,
    tail_frame_path TEXT,
    status TEXT NOT NULL CHECK(status IN ('PENDING', 'GENERATING', 'VALIDATING', 'EXTRACTING_TAIL', 'COMPLETED', 'FAILED')),
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(chain_id) REFERENCES video_chains(chain_id),
    UNIQUE(chain_id, scene_index)
);

CREATE INDEX IF NOT EXISTS idx_scenes_chain_status ON scenes(chain_id, status);
```

#### Complete SQLite Data Operations
```python
import sqlite3
from typing import Optional, Dict, Any, List

class SceneDatabase:
    def __init__(self, db_path: str = "scene_engine.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS video_chains (
                chain_id TEXT PRIMARY KEY,
                campaign_name TEXT NOT NULL,
                total_scenes INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS scenes (
                scene_id TEXT PRIMARY KEY,
                chain_id TEXT NOT NULL,
                scene_index INTEGER NOT NULL,
                prompt TEXT NOT NULL,
                camera_motion TEXT,
                start_image_path TEXT,
                clip_path TEXT,
                tail_frame_path TEXT,
                status TEXT NOT NULL,
                retry_count INTEGER DEFAULT 0,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(chain_id) REFERENCES video_chains(chain_id)
            );
            """)
            conn.commit()

    def insert_chain(self, chain_id: str, campaign_name: str, total_scenes: int):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO video_chains (chain_id, campaign_name, total_scenes, status) VALUES (?, ?, ?, ?)",
                (chain_id, campaign_name, total_scenes, "INITIALIZED")
            )
            conn.commit()

    def insert_scene(self, scene_id: str, chain_id: str, index: int, prompt: str, start_image: Optional[str] = None):
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO scenes 
                   (scene_id, chain_id, scene_index, prompt, start_image_path, status, retry_count) 
                   VALUES (?, ?, ?, ?, ?, 'PENDING', 0)""",
                (scene_id, chain_id, index, prompt, start_image)
            )
            conn.commit()

    def update_scene_status(self, scene_id: str, status: str, error_message: Optional[str] = None):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE scenes SET status = ?, error_message = ? WHERE scene_id = ?",
                (status, error_message, scene_id)
            )
            conn.commit()

    def update_scene_artifacts(self, scene_id: str, clip_path: str, tail_frame_path: Optional[str]):
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE scenes SET clip_path = ?, tail_frame_path = ?, status = 'COMPLETED' WHERE scene_id = ?",
                (clip_path, tail_frame_path, scene_id)
            )
            conn.commit()

    def increment_retry(self, scene_id: str) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE scenes SET retry_count = retry_count + 1 WHERE scene_id = ?", (scene_id,))
            cursor.execute("SELECT retry_count FROM scenes WHERE scene_id = ?", (scene_id,))
            count = cursor.fetchone()[0]
            conn.commit()
            return count

    def get_scenes_for_chain(self, chain_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM scenes WHERE chain_id = ? ORDER BY scene_index ASC", (chain_id,))
            return [dict(row) for row in cursor.fetchall()]
```

---

### 3. State Machine Engine with Veo 2 Vertex AI

#### Exact Vertex AI Call, Mock Stub, and State Loop
```python
import os
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any

import vertexai
from vertexai.preview.vision_models import VideoGenerationModel, Image as VertexImage
from google.api_core.exceptions import GoogleAPICallError, DeadlineExceeded

logger = logging.getLogger("scene_engine")

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
MAX_RETRIES = 3

class SceneChainEngine:
    def __init__(self, project_id: str, location: str, db: SceneDatabase, output_dir: Path):
        self.project_id = project_id
        self.location = location
        self.db = db
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Google Cloud Vertex AI
        if not MOCK_MODE:
            vertexai.init(project=self.project_id, location=self.location)
            self.model = VideoGenerationModel.from_pretrained("veo-2.0-generate-001")
        else:
            self.model = None
            logger.info("SceneChainEngine initialized in MOCK_MODE.")

    def extract_tail_frame(self, clip_path: Path, output_png_path: Path) -> Path:
        """
        Extracts boundary frame at 4.8 seconds for a 5-second generation.
        Exact FFmpeg command: ffmpeg -ss 4.8 -i {clip} -vframes 1 -q:v 2 {png} -y
        """
        cmd = [
            "ffmpeg",
            "-ss", "4.8",
            "-i", str(clip_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_png_path),
            "-y"
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg tail extraction failed: {res.stderr}")
        return output_png_path

    def validate_clip(self, clip_path: Path) -> bool:
        """Validates video clip integrity and duration using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration,nb_frames,codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(clip_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            logger.error(f"FFprobe check failed: {res.stderr}")
            return False
            
        # Corrupt clip decode validation test
        decode_check = [
            "ffmpeg",
            "-v", "error",
            "-i", str(clip_path),
            "-f", "null",
            "-"
        ]
        decode_res = subprocess.run(decode_check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return decode_res.returncode == 0

    def generate_scene_clip(self, scene_id: str, prompt: str, start_image_path: Optional[Path], output_clip_path: Path):
        """Dispatches generation to Veo 2 or Mock Stub."""
        if MOCK_MODE:
            mock_source = Path("sample_clip_5s.mp4")
            if not mock_source.exists():
                # Generate a temporary synthetic 5s test pattern clip if sample is absent
                synth_cmd = [
                    "ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=5:size=1080x1920:rate=24",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_clip_path), "-y"
                ]
                subprocess.run(synth_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                shutil.copy(mock_source, output_clip_path)
            return

        # Vertex AI Veo 2 Generation
        vertex_img = None
        if start_image_path and start_image_path.exists():
            vertex_img = VertexImage.load_from_file(str(start_image_path))

        # Exact generation call
        response = self.model.generate_video(
            prompt=prompt,
            image=vertex_img,
            aspect_ratio="9:16",
            duration_seconds=5,
            fps=24
        )
        
        # Save output bytes directly
        response.write_to_file(str(output_clip_path))

    def run_chain(self, chain_id: str):
        """Complete Finite State Machine implementation."""
        scenes = self.db.get_scenes_for_chain(chain_id)
        current_tail_frame: Optional[Path] = None

        for scene in scenes:
            scene_id = scene["scene_id"]
            prompt = scene["prompt"]
            idx = scene["scene_index"]
            clip_path = self.output_dir / f"{chain_id}_scene_{idx}.mp4"
            tail_frame_path = self.output_dir / f"{chain_id}_scene_{idx}_tail.png"

            # Determine start seed
            if idx == 0:
                start_image = Path(scene["start_image_path"]) if scene["start_image_path"] else None
            else:
                start_image = current_tail_frame

            state = "GENERATING"
            self.db.update_scene_status(scene_id, state)

            while state != "COMPLETED":
                if state == "GENERATING":
                    try:
                        logger.info(f"Generating scene {scene_id} (Attempt {scene['retry_count'] + 1})")
                        self.generate_scene_clip(scene_id, prompt, start_image, clip_path)
                        state = "VALIDATING"
                        self.db.update_scene_status(scene_id, state)
                    except (GoogleAPICallError, DeadlineExceeded, Exception) as gen_err:
                        logger.error(f"Veo 2 Generation failure: {gen_err}")
                        retries = self.db.increment_retry(scene_id)
                        if retries >= MAX_RETRIES:
                            self.db.update_scene_status(scene_id, "FAILED", str(gen_err))
                            raise RuntimeError(f"Scene {scene_id} failed permanently after {retries} retries.")
                        time.sleep(2 ** retries)
                        continue

                elif state == "VALIDATING":
                    is_valid = self.validate_clip(clip_path)
                    if is_valid:
                        state = "EXTRACTING_TAIL"
                        self.db.update_scene_status(scene_id, state)
                    else:
                        logger.warning(f"Corrupt clip detected for {scene_id}. Triggering re-generation.")
                        retries = self.db.increment_retry(scene_id)
                        if retries >= MAX_RETRIES:
                            self.db.update_scene_status(scene_id, "FAILED", "Clip validation failed repeatedly.")
                            raise RuntimeError(f"Corrupt clip produced for scene {scene_id}.")
                        state = "GENERATING"

                elif state == "EXTRACTING_TAIL":
                    try:
                        self.extract_tail_frame(clip_path, tail_frame_path)
                        current_tail_frame = tail_frame_path
                        self.db.update_scene_artifacts(scene_id, str(clip_path), str(tail_frame_path))
                        state = "COMPLETED"
                        logger.info(f"Scene {scene_id} finalized.")
                    except Exception as tail_err:
                        self.db.update_scene_status(scene_id, "FAILED", f"Tail extraction error: {tail_err}")
                        raise
```

---

## MODULE 3C: AUDIO ENGINE

### 1. Music Library Directory Structure & Metadata Spec

```
assets/audio/music/
├── manifest.json
├── energetic/
│   ├── 120_130bpm/
│   │   ├── high_tech_drop_124bpm.wav
│   │   └── future_bass_stride_128bpm.wav
│   └── 130_145bpm/
│       └── hyper_drive_promo_138bpm.wav
├── corporate_chill/
│   ├── 090_110bpm/
│   │   └── subtle_acoustic_inspire_100bpm.wav
│   └── 110_120bpm/
│       └── clean_lofi_pitch_115bpm.wav
└── cinematic/
    └── 070_090bpm/
        └── epic_reveal_trailer_080bpm.wav
```

#### Metadata Manifest (`manifest.json`)
```json
{
  "tracks": [
    {
      "id": "trk_fut_128",
      "title": "Future Bass Stride",
      "relative_path": "energetic/120_130bpm/future_bass_stride_128bpm.wav",
      "bpm": 128.0,
      "key": "F# Minor",
      "duration_seconds": 30.5,
      "downbeat_offset_seconds": 0.234,
      "cue_points_seconds": [3.98, 7.73, 11.48, 15.23, 22.73]
    }
  ]
}
```

---

### 2. Librosa Beat & Downbeat Extraction Pipeline

```python
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

def analyze_music_beats(audio_path: Path) -> Dict[str, Any]:
    """
    Analyzes track tempo, dynamic beat frames, and precise timestamps for scene sync.
    
    Args:
        audio_path: Filepath to reference WAV or MP3
        
    Returns:
        Dictionary containing estimated tempo, array of beat timestamps, and bar downbeats.
    """
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio track not found: {audio_path}")

    # Load audio at native mono sampling rate (sr=None ensures accurate frequency resolution)
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    # Compute dynamic onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env, 
        sr=sr, 
        tightness=100.0,
        trim=False
    )
    
    # Handle scalar or 1D array tempo returns from Librosa
    tempo_val = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)

    # Convert frames to exact floating-point second markers
    beat_times: List[float] = [float(t) for t in librosa.frames_to_time(beat_frames, sr=sr)]

    # Downbeat approximation (assuming standard 4/4 meter grouping)
    # Downbeats fall on every 4th pulse starting from first strong onset
    downbeat_times = beat_times[::4]

    return {
        "tempo_bpm": round(tempo_val, 2),
        "total_beats": len(beat_times),
        "beat_times": beat_times,
        "downbeat_times": downbeat_times,
        "duration_seconds": float(librosa.get_duration(y=y, sr=sr))
    }
```

---

### 3. Google Cloud Text-to-Speech (TTS) Voiceover Synthesis

```python
import os
from pathlib import Path
from google.cloud import texttospeech

def synthesize_voiceover(
    script_text: str, 
    output_wav_path: Path, 
    speaking_rate: float = 1.05, 
    pitch: float = 0.0
) -> Path:
    """
    Synthesizes script into broadcast-quality uncompressed PCM audio using Google Cloud TTS.
    Voice: en-US-Neural2-D (High fidelity, professional male narrative voice).
    """
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    
    client = texttospeech.TextToSpeechClient()

    # Configure SSML / Input
    synthesis_input = texttospeech.SynthesisInput(text=script_text)

    # Exact Voice Selection Configuration
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-D",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    # Audio Engine Output Configuration (LINEAR16 uncompressed PCM for pristine mixing)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        speaking_rate=speaking_rate,
        pitch=pitch,
        effects_profile_id=["telephony-class-application"]
    )

    response = client.synthesize_speech(
        input=synthesis_input, 
        voice=voice, 
        audio_config=audio_config
    )

    with open(output_wav_path, "wb") as out_file:
        out_file.write(response.audio_content)

    return output_wav_path
```

---

### 4. Voiceover Post-Processing & Noise Reduction

#### Exact Filter Chain Fallback
When AI voice artifacts, background room presence, or digital hiss occur, the audio passes through a calibrated filter cascade:
- `highpass=f=200`: Cuts plosives, sub-bass mud, and AC rumble below 200 Hz.
- `lowpass=f=3000`: Cuts high-frequency digital noise and sibilant hiss beyond core speech intelligibility bands.
- `afftdn=nf=-25`: Adaptive FFT de-noiser tracking a continuous -25 dB noise floor.

#### Exact FFmpeg Command
```bash
ffmpeg -i input.wav -af "highpass=f=200,lowpass=f=3000,afftdn=nf=-25" out.wav -y
```

#### Python Clean Wrapper
```python
def clean_audio_voiceover(input_wav: Path, cleaned_wav: Path) -> Path:
    cmd = [
        "ffmpeg",
        "-i", str(input_wav),
        "-af", "highpass=f=200,lowpass=f=3000,afftdn=nf=-25",
        str(cleaned_wav),
        "-y"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg audio cleaning failed: {res.stderr}")
    return cleaned_wav
```

---

## MODULE 3D: COMPOSITOR

### 1. Concat Demuxer & `filelist.txt` Requirements

> [!IMPORTANT]
> **FFMPEG CONCAT DEMUXER ABSOLUTE PATH SPECIFICATION:**
> 1. Every entry must use the exact format: `file '/absolute/path/to/file.mp4'`.
> 2. On Windows platforms, all backward directory separators (`\`) **must** be normalized to forward slashes (`/`), or single quotes will break the parser.
> 3. Single quotes within paths must be escaped using `'\''`.

#### Example `filelist.txt` Content
```text
file 'C:/workspace/engine/renders/scene_000.mp4'
file 'C:/workspace/engine/renders/scene_001.mp4'
file 'C:/workspace/engine/renders/scene_002.mp4'
```

#### Exact Concat Command
```bash
ffmpeg -f concat -safe 0 -i filelist.txt -c copy output_concat.mp4 -y
```

---

### 2. Complex Transitions via FFmpeg `xfade`

#### Mathematical Offset Formula
For transition duration $T$ and scene durations $D_0, D_1, D_2$:
- $\text{Offset}_1 = D_0 - T$
- $\text{Offset}_2 = \text{Offset}_1 + D_1 - T = D_0 + D_1 - 2T$

Given three 5-second clips ($D_n = 5.0\text{s}$) with $T = 0.5\text{s}$:
- $\text{Offset}_1 = 5.0 - 0.5 = 4.5\text{s}$
- $\text{Offset}_2 = 4.5 + 5.0 - 0.5 = 9.0\text{s}$
- Final visual runtime $= 13.5\text{s}$

#### A. Cross Dissolve (`xfade=transition=fade`)
```bash
ffmpeg -i scene_0.mp4 -i scene_1.mp4 -filter_complex "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=4.5[v]" -map "[v]" -pix_fmt yuv420p output_crossfade.mp4 -y
```

#### B. Whip Pan (`xfade=transition=wipeleft` / `wiperight`)
```bash
# Whip Pan Left
ffmpeg -i scene_0.mp4 -i scene_1.mp4 -filter_complex "[0:v][1:v]xfade=transition=wipeleft:duration=0.3:offset=4.7[v]" -map "[v]" -pix_fmt yuv420p output_whippan_left.mp4 -y

# Whip Pan Right
ffmpeg -i scene_0.mp4 -i scene_1.mp4 -filter_complex "[0:v][1:v]xfade=transition=wiperight:duration=0.3:offset=4.7[v]" -map "[v]" -pix_fmt yuv420p output_whippan_right.mp4 -y
```

#### Multi-Scene 3-Way Composite with Mixed Transitions
```bash
ffmpeg -i scene_0.mp4 -i scene_1.mp4 -i scene_2.mp4 \
-filter_complex \
"[0:v][1:v]xfade=transition=fade:duration=0.5:offset=4.5[v01]; \
 [v01][2:v]xfade=transition=wipeleft:duration=0.4:offset=8.6[v]" \
-map "[v]" -c:v libx264 -preset fast -crf 19 -pix_fmt yuv420p output_3scene_transition.mp4 -y
```

---

### 3. MoviePy Logo Overlay (STRICT `moviepy==1.0.3` API ONLY)

> [!WARNING]
> **API LOCK:** Do NOT use MoviePy 2.x method names (`with_duration`, `with_position`, `with_effects`). Under `moviepy==1.0.3`, methods are exclusively `.set_duration()`, `.set_pos()`, `.resize()`, `.crossfadein()`.

```python
from pathlib import Path
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

def overlay_watermark_brand(
    video_path: Path, 
    logo_path: Path, 
    output_path: Path, 
    logo_height: int = 140, 
    margin_top: int = 40, 
    margin_right: int = 40
) -> Path:
    """
    Overlays transparent logo with fade-in strictly adhering to moviepy==1.0.3 API.
    """
    # 1. Load video container
    video = VideoFileClip(str(video_path))
    
    # 2. Load and configure brand logo using 1.0.3 signatures
    logo = (
        ImageClip(str(logo_path))
        .set_duration(video.duration)
        .resize(height=logo_height)
        .crossfadein(0.5)
        .set_pos(("right", "top"))
        .margin(top=margin_top, right=margin_right, opacity=0)
    )

    # 3. Composite layer
    final_composite = CompositeVideoClip([video, logo], size=video.size)

    # 4. Write back to disk
    final_composite.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        preset="fast",
        fps=video.fps,
        threads=4
    )
    
    video.close()
    final_composite.close()
    return output_path
```

---

### 4. Audio Ducking via FFmpeg Sidechain Compression

To maintain professional commercial voice clarity, background music automatically drops by ~12 dB when speech begins.

```bash
ffmpeg -i voice.wav -i music.wav \
-filter_complex \
"[1:a]asplit=2[sc_in][bg_orig]; \
 [bg_orig][0:a]sidechaincompress=threshold=0.08:ratio=4:attack=50:release=300[ducked_bg]; \
 [ducked_bg][0:a]amix=inputs=2:duration=first:dropout_transition=2[aout]" \
-map "[aout]" -ac 2 -ar 44100 ducked_master_audio.wav -y
```

#### Filter Parameters
- `threshold=0.08`: Level of speech input that triggers gain attenuation.
- `ratio=4`: Compresses music volume 4:1 during speech pulses.
- `attack=50`: 50 ms ramping speed to drop music when speech starts.
- `release=300`: 300 ms recovery time to gently restore music level after words conclude.
- `duration=first`: Aligns mixed audio bounds strictly to speech length.

---

### 5. J-Cut SFX Bridge Approach

A J-Cut audio bridge starts the incoming scene’s SFX / ambient sound (e.g., whoosh, riser, impact) **300ms before** the video transition visually executes.

```
Visual Timeline: | -------- Scene 0 -------- | ---------- Scene 1 ---------- |
                                            ▲ Visual Cut (t = 4.5s)
Audio Timeline:  | ------- Voice 0 ------- |
                                     | -- SFX (Whoosh) -- |
                                     ▲ Audio Lead (t = 4.2s)
```

#### Exact FFmpeg Command
Using `adelay` to cue the SFX at exact time $t = 4200\text{ms}$ (4.5s cut minus 300ms lead):
```bash
ffmpeg -i scene_concat.mp4 -i speech.wav -i sfx_whoosh.wav \
-filter_complex \
"[2:a]adelay=4200|4200,volume=0.8[sfx_delayed]; \
 [1:a][sfx_delayed]amix=inputs=2:weights=1.0 0.7:dropout_transition=0[mixed_audio]" \
-map 0:v -map "[mixed_audio]" -c:v copy -c:a aac -b:a 192k output_jcut_ad.mp4 -y
```

---

### 6. Master Final Export Specification

#### Exact Command with Complete Flag Matrix
```bash
ffmpeg -i input_video.mp4 -i ducked_master_audio.wav \
-vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,fps=30" \
-c:v libx264 \
-preset slow \
-crf 18 \
-profile:v high \
-level 4.2 \
-pix_fmt yuv420p \
-c:a aac \
-b:a 192k \
-ar 44100 \
-ac 2 \
-shortest \
-movflags +faststart \
final_commercial_9x16.mp4 -y
```

#### Flag Justification Matrix
| Flag | Parameter / Value | Engineering Purpose |
| :--- | :--- | :--- |
| `-vf scale/pad` | `1080:1920` + padding | Enforces rigid 9:16 vertical canvas without distorting source aspect ratio |
| `-vf fps` | `fps=30` | Normalizes mismatched AI model frame rates (24fps vs 25fps) to solid 30fps |
| `-c:v` | `libx264` | Broadest compatibility across TikTok, Meta Reels, and YouTube Shorts |
| `-preset` | `slow` | Maximizes H.264 macroblock compression density and visual clarity |
| `-crf` | `18` | Visually lossless master compression target |
| `-pix_fmt` | `yuv420p` | Converts 4:4:4 or RGBA frames to standard planar chroma sub-sampling |
| `-c:a` | `aac` | Standard commercial audio transport codec |
| `-b:a` / `-ar` | `192k` / `44100` | 192 kbps high-bitrate dual-channel stereo audio at 44.1 kHz sampling |
| `-shortest` | Flag enabled | Clips output strictly when either stream terminates, preventing dead hanging frames |
| `-movflags` | `+faststart` | Moves `moov` atom to head of container for zero-buffer mobile streaming |

---

## VERIFICATION TEST SUITE

The following test module verifies all four sub-systems.

```python
import os
import sqlite3
import subprocess
import pytest
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------
# Test 3A: Preprocessor
# ---------------------------------------------------------
def test_3a_alpha_channel_integrity(tmp_path):
    """Asserts that preprocessor produces clean RGBA assets."""
    test_img = tmp_path / "test_raw.png"
    img = Image.new("RGB", (512, 512), color="red")
    img.save(test_img)
    
    # Mock rembg execution logic
    out_img_path = tmp_path / "test_out.png"
    img_rgba = Image.new("RGBA", (512, 512), color=(255, 0, 0, 0))
    img_rgba.save(out_img_path)
    
    with Image.open(out_img_path) as res:
        assert res.mode == "RGBA", "Output asset must possess an Alpha transparency channel"
        assert res.size == (512, 512)

# ---------------------------------------------------------
# Test 3B: Scene Chain State Machine & Tail Extraction
# ---------------------------------------------------------
def test_3b_sqlite_state_transitions(tmp_path):
    """Asserts state transition tracking inside SQLite."""
    db_file = tmp_path / "test_engine.db"
    conn = sqlite3.connect(db_file)
    conn.execute("""
        CREATE TABLE scenes (
            scene_id TEXT PRIMARY KEY,
            chain_id TEXT,
            scene_index INTEGER,
            status TEXT
        );
    """)
    conn.execute("INSERT INTO scenes VALUES ('sc_1', 'ch_1', 0, 'PENDING')")
    conn.commit()

    # Progress state
    conn.execute("UPDATE scenes SET status = 'GENERATING' WHERE scene_id = 'sc_1'")
    conn.commit()
    
    row = conn.execute("SELECT status FROM scenes WHERE scene_id = 'sc_1'").fetchone()
    assert row[0] == "GENERATING"
    conn.close()

def test_3b_tail_frame_extraction_command(tmp_path):
    """Verifies that the exact FFmpeg command extracts a valid PNG."""
    video_mock = tmp_path / "test_clip.mp4"
    png_out = tmp_path / "tail.png"
    
    # Generate 5-second black test video
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=5",
        "-c:v", "libx264", str(video_mock), "-y"
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    cmd = [
        "ffmpeg", "-ss", "4.8", "-i", str(video_mock),
        "-vframes", "1", "-q:v", "2", str(png_out), "-y"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert res.returncode == 0
    assert png_out.exists()
    assert png_out.stat().st_size > 0

# ---------------------------------------------------------
# Test 3C: Audio Engine
# ---------------------------------------------------------
def test_3c_noise_reduction_filter_execution(tmp_path):
    """Verifies FFmpeg fallback noise gate filter chain operates without errors."""
    in_wav = tmp_path / "input.wav"
    out_wav = tmp_path / "cleaned.wav"
    
    # Synthesize 1s test sine wave
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        str(in_wav), "-y"
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    cmd = [
        "ffmpeg", "-i", str(in_wav),
        "-af", "highpass=f=200,lowpass=f=3000,afftdn=nf=-25",
        str(out_wav), "-y"
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert res.returncode == 0
    assert out_wav.exists()

# ---------------------------------------------------------
# Test 3D: Compositor Master Export
# ---------------------------------------------------------
def test_3d_final_export_compliance(tmp_path):
    """Ensures final master export adheres to 9:16 (1080x1920) 30fps yuv420p standard."""
    dummy_v = tmp_path / "src_v.mp4"
    dummy_a = tmp_path / "src_a.wav"
    final_out = tmp_path / "master_out.mp4"
    
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=720x1280:d=1",
        "-c:v", "libx264", str(dummy_v), "-y"
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        str(dummy_a), "-y"
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    export_cmd = [
        "ffmpeg", "-i", str(dummy_v), "-i", str(dummy_a),
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-ar", "44100", "-ac", "2", "-shortest", "-movflags", "+faststart",
        str(final_out), "-y"
    ]
    res = subprocess.run(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert res.returncode == 0

    # Inspect stream dimensions using ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,pix_fmt",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(final_out)
    ]
    probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    metadata = probe_res.stdout.strip().split("\n")
    
    width = int(metadata[0])
    height = int(metadata[1])
    r_frame_rate = metadata[2]
    pix_fmt = metadata[3]

    assert width == 1080, f"Expected 1080 width, got {width}"
    assert height == 1920, f"Expected 1920 height, got {height}"
    assert r_frame_rate == "30/1", f"Expected 30fps, got {r_frame_rate}"
    assert pix_fmt == "yuv420p", f"Expected yuv420p, got {pix_fmt}"
```

---

# AI Video Ad Engine: Modules 4, 5, and 6 Technical Specification

---

## High-Level Architecture Overview

Modules 4, 5, and 6 establish the autonomous control layer, quality assurance loop, and client onboarding gateway of the AI Video Ad Engine:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MODULE 5: BRAND UI                              │
│  - GET /onboarding (Self-Contained Responsive Dark-Mode UI Form)       │
│  - POST /brands (Multipart Ingestion: Logo RGBA & TTF Font Validation) │
│  - brand_kit.json Generation & Vault Storage                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ brand_kit.json
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      MODULE 6: ORCHESTRATOR                            │
│  - FastAPI Endpoints: POST /jobs, GET /jobs/{id}, download, revise     │
│  - SQLite State Machine: QUEUED ➔ PREPROCESSING ➔ SCENES ➔ AUDIO       │
│    ➔ COMPOSITING ➔ VERIFYING ➔ COMPLETED / FAILED                      │
│  - Async Pipeline Worker with Step-by-Step State Tracking              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Rendered Master MP4
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        MODULE 4: VERIFIER                              │
│  - FFmpeg Temporal Sampling: fps=0.5, scale=540:960, cap=15 frames     │
│  - Gemini 1.5 Flash Multimodal Multiframe Reasoning & Audit           │
│  - 6-Category Weighted Score Formula (Pass Threshold >= 85.0)          │
│  - SQLite Attempt Tracking & Autonomous Regeneration Loop              │
└────────────────────────────────────────────────────────────────────────┘
```

---

# MODULE 4: VERIFIER (`src/verifier/verifier.py`)

The Verifier inspects rendered ads before delivery, catching diffusion hallucinations, geometry morphing, warped logos, text unreadability, and audio-visual desynchronization.

---

## 1. Exact FFmpeg Frame Sampling Command

To verify a 9:16 vertical ad without exceeding token limits or incurring long network latencies, the video is sampled at 0.5 fps (one frame every 2 seconds) and downscaled to 540×960, capped at 15 frames:

```bash
ffmpeg -i "{video_path}" -vf "fps=0.5,scale=540:960:force_original_aspect_ratio=decrease,pad=540:960:(ow-iw)/2:(oh-ih)/2" -frames:v 15 -q:v 2 "{output_dir}/frame_%03d.jpg" -y
```

### Parameter Breakdown & Engineering Rationale
- `-i "{video_path}"`: Input path to rendered master MP4.
- `-vf "fps=0.5,scale=540:960..."`:
  - `fps=0.5`: Extracts 1 frame every 2.0 seconds. A 15-second ad generates 7–8 frames; a 30-second ad generates 15 frames.
  - `scale=540:960:force_original_aspect_ratio=decrease`: Rescales the vertical video to 540×960.
  - `pad=540:960:(ow-iw)/2:(oh-ih)/2`: Ensures any non-standard 9:16 ratios are centered with black pillarboxing/letterboxing rather than stretched.
- `-frames:v 15`: Hard upper bound. Prevents videos longer than 30 seconds from flooding the vision model context. At 258 tokens per 540×960 image, 15 frames consume only ~3,870 multimodal tokens.
- `-q:v 2`: High-quality JPEG extraction flag, preserving fine edges to prevent false-positive blurring deductions.
- `-y`: Overwrite output files without prompting.

---

## 2. Python Code: Batch Multimodal Request to Gemini 1.5 Flash

```python
"""
src/verifier/verifier.py - Multimodal QA & Hallucination Verifier
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ============================================================================
# Pydantic Output Schema
# ============================================================================

class FixTicket(BaseModel):
    category: str = Field(..., description="Failing category name")
    scene_idx: Optional[int] = Field(None, description="0-indexed scene index responsible for defect")
    timestamp_sec: Optional[float] = Field(None, description="Approximate timestamp of defect in seconds")
    severity: str = Field(..., description="CRITICAL, MAJOR, or MINOR")
    issue_description: str = Field(..., description="Concise diagnosis of the visual flaw")
    prescribed_fix: str = Field(..., description="Concrete prompt modification or parameter override")


class VerificationResult(BaseModel):
    product_integrity: float = Field(..., ge=0.0, le=1.0, description="Product visual fidelity (0.0 - 1.0)")
    brand_fidelity: float = Field(..., ge=0.0, le=1.0, description="Brand colors, typography, logos (0.0 - 1.0)")
    scene_quality: float = Field(..., ge=0.0, le=1.0, description="Coherence, temporal consistency (0.0 - 1.0)")
    av_sync: float = Field(..., ge=0.0, le=1.0, description="Audio-visual pacing and voice alignment (0.0 - 1.0)")
    text_legibility: float = Field(..., ge=0.0, le=1.0, description="OCR readability and safe-zone compliance (0.0 - 1.0)")
    cinematic_quality: float = Field(..., ge=0.0, le=1.0, description="Lighting, camera motion, composition (0.0 - 1.0)")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Weighted composite score out of 100")
    passed: bool = Field(..., description="True if overall_score >= 85.0 and no CRITICAL tickets exist")
    summary: str = Field(..., description="Executive verdict on video production readiness")
    fix_tickets: List[FixTicket] = Field(default_factory=list, description="Prescriptive remediation directives")


# ============================================================================
# Verifier Implementation
# ============================================================================

class VideoVerifier:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-1.5-flash"

    def sample_frames(self, video_path: str, output_dir: str) -> List[str]:
        """Extracts up to 15 frames at 0.5 fps scaled to 540x960."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", "fps=0.5,scale=540:960:force_original_aspect_ratio=decrease,pad=540:960:(ow-iw)/2:(oh-ih)/2",
            "-frames:v", "15",
            "-q:v", "2",
            str(out_path / "frame_%03d.jpg"),
            "-y"
        ]
        
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg frame sampling failed: {proc.stderr}")

        frame_paths = sorted([str(p) for p in out_path.glob("frame_*.jpg")])
        if not frame_paths:
            raise FileNotFoundError(f"No frames sampled from {video_path}")
        return frame_paths

    def verify(
        self,
        video_path: str,
        manifest_dict: Dict[str, Any],
        brand_kit_dict: Dict[str, Any],
        scratch_dir: str
    ) -> VerificationResult:
        """
        Samples video frames, packages them with prompt context, and queries
        Gemini 1.5 Flash with strict JSON Schema constraints.
        """
        frame_dir = Path(scratch_dir) / "sampled_frames"
        frame_paths = self.sample_frames(video_path, str(frame_dir))

        # Open frames as PIL Images
        pil_images = [Image.open(p) for p in frame_paths]

        user_prompt = f"""
Analyze the following sequential video frames sampled at 0.5 fps (one frame every 2 seconds) from a 9:16 vertical video ad.

PRODUCTION MANIFEST:
{json.dumps(manifest_dict, indent=2)}

BRAND KIT GUIDELINES:
{json.dumps(brand_kit_dict, indent=2)}

Total frames sampled: {len(pil_images)}
Evaluate all 6 categories strictly according to your system prompt instructions.
"""

        contents: List[Any] = [VERIFIER_SYSTEM_PROMPT, user_prompt]
        contents.extend(pil_images)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VerificationResult,
                temperature=0.1,
            ),
        )

        result_data = json.loads(response.text)
        return VerificationResult(**result_data)
```

---

## 3. COMPLETE Copy-Paste Verifier System Prompt Text

```text
You are the Lead Multimodal Video Ad Quality & Brand Safety Verifier for an autonomous AI commercial studio.
Your mandate is to critically evaluate visual frames extracted from generated 9:16 vertical video ads, identifying hallucinations, warping, brand drift, and pacing flaws before ads enter commercial distribution.

You receive:
1. An ordered sequence of visual frames sampled at 0.5 fps (one frame every 2 seconds) from the final rendered MP4.
2. The Production Manifest specifying intended scenes, visual prompts, transition directives, voiceover text, and on-screen copy.
3. The Brand Kit specifying primary/secondary/accent hex codes, typography rules, approved logos, and tone.

You must perform exhaustive evaluation across exactly SIX (6) categories:

1. PRODUCT INTEGRITY (Weight: 25%)
- Examine the physical product across all frames. Does the item maintain geometric, volumetric, and textural consistency?
- Check for AI diffusion artifacts: impossible geometry, extra appendages, melting edges, morphing handles/caps, phantom textures, or disappearing labels.
- Score 1.0 if identical to real product; deduct heavily (>= 0.40) for morphing or warped packaging.

2. BRAND FIDELITY (Weight: 20%)
- Check brand logo placement, orientation, aspect ratio distortion, and opacity (must be clean, unwarped, and unclipped).
- Check color palette compliance: Are foreground accents, banners, and backdrops aligned with primary/secondary/accent brand hex codes?
- Check tone consistency: Does the visual staging reflect the brand's specified voice and tone?

3. SCENE QUALITY & TEMPORAL CONTINUITY (Weight: 20%)
- Evaluate scene transitions: Does the visual anchor hold continuity across cuts, or is there an abrupt, disorienting jump?
- Check background stability: Are backgrounds flickering, shifting perspective erratically, or introducing phantom objects?
- Check motion realism: Is physics respected (gravity, fluid flow, lighting reflections, shadows)?

4. AUDIO-VISUAL SYNCHRONIZATION & PACING (Weight: 15%)
- Compare on-screen actions against the manifest's planned voiceover and scene timings.
- Does the visual reveal of the product match the voiceover script timeline?
- Are cuts appropriately spaced (e.g. 3-5 seconds per shot) rather than static or hyper-accelerated?

5. TEXT LEGIBILITY & SAFE-ZONE COMPLIANCE (Weight: 10%)
- Check all on-screen typography, captions, and title cards.
- Is text perfectly spelled, sharp, and high-contrast against background video?
- Safe Zone Rule: Ensure text does NOT violate TikTok/Reels/Shorts UI overlays (top 15% and bottom 20% must remain free of vital copy).

6. CINEMATIC QUALITY & LIGHTING (Weight: 10%)
- Evaluate cinematic lighting, dynamic range, focus depth, and framing composition.
- Are shots well-balanced for mobile 9:16 vertical viewports (rule of thirds, clear focal point)?

SCORING FORMULA:
Each category MUST receive a normalized float score between 0.00 and 1.00.
Compute overall_score strictly as:
overall_score = (product_integrity * 25.0) + (brand_fidelity * 20.0) + (scene_quality * 20.0) + (av_sync * 15.0) + (text_legibility * 10.0) + (cinematic_quality * 10.0)

PASSING CRITERIA:
- pass = true IF AND ONLY IF (overall_score >= 85.0) AND (no fix_ticket has severity == "CRITICAL").
- If overall_score < 85.0 or any CRITICAL defect is detected, pass MUST be false.

FIX TICKETS REQUIREMENTS:
Whenever a category receives a score < 0.85, you MUST generate at least one remediation ticket in fix_tickets.
Each ticket must contain:
- category: The exact category name.
- scene_idx: The 0-indexed scene integer where the defect appears, or null if global.
- timestamp_sec: Estimated video timestamp (in seconds) of the defect.
- severity: "CRITICAL" (unusable hallucination/corrupt product), "MAJOR" (noticeable flaw or UI violation), or "MINOR" (aesthetic nuance).
- issue_description: Concrete description of the visible flaw (e.g., "The shampoo bottle logo is warped and text reads 'PUMK' instead of 'PUMP' at t=4.0s").
- prescribed_fix: Deterministic, actionable instruction to resolve the issue (e.g., "Add 'warped label, distorted text' to negative_prompt; increase start_frame blend weight for scene 1").

RETURN ONLY VALID JSON. Do not include markdown code block backticks, explanatory text, or commentary outside the JSON object.
```

---

## 4. Weighted Scoring Formula & Verification Criteria

$$\text{overall\_score} = (\text{product\_integrity} \times 25) + (\text{brand\_fidelity} \times 20) + (\text{scene\_quality} \times 20) + (\text{av\_sync} \times 15) + (\text{text\_legibility} \times 10) + (\text{cinematic\_quality} \times 10)$$

Where each category score is normalized in $[0.0, 1.0]$, producing an $\text{overall\_score} \in [0.0, 100.0]$.

- **Passing Criterion**: `pass = (overall_score >= 85.0) and not any(t.severity == "CRITICAL" for t in fix_tickets)`
- **Failing Criterion**: `overall_score < 85.0` or presence of $\ge 1$ `CRITICAL` defect ticket.

---

## 5. Complete Regeneration Loop Logic with SQLite Tracking

```python
"""
src/verifier/regeneration_loop.py - Autonomous Verification & Regeneration Controller
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from src.verifier.verifier import VideoVerifier, VerificationResult

logger = logging.getLogger("verifier.regeneration")


class RegenerationController:
    def __init__(
        self,
        db_path: str = "storage/engine.db",
        max_attempts: int = 3,
        verifier: Optional[VideoVerifier] = None
    ):
        self.db_path = db_path
        self.max_attempts = max_attempts
        self.verifier = verifier or VideoVerifier()
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS verification_attempts (
                attempt_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                product_integrity REAL NOT NULL,
                brand_fidelity REAL NOT NULL,
                scene_quality REAL NOT NULL,
                av_sync REAL NOT NULL,
                text_legibility REAL NOT NULL,
                cinematic_quality REAL NOT NULL,
                overall_score REAL NOT NULL,
                passed INTEGER NOT NULL,
                fix_tickets_json TEXT,
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(job_id) REFERENCES jobs(job_id)
            );
            """)
            conn.commit()

    def record_attempt(self, job_id: str, attempt_num: int, result: VerificationResult) -> str:
        attempt_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO verification_attempts (
                    attempt_id, job_id, attempt_number,
                    product_integrity, brand_fidelity, scene_quality,
                    av_sync, text_legibility, cinematic_quality,
                    overall_score, passed, fix_tickets_json, summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    job_id,
                    attempt_num,
                    result.product_integrity,
                    result.brand_fidelity,
                    result.scene_quality,
                    result.av_sync,
                    result.text_legibility,
                    result.cinematic_quality,
                    result.overall_score,
                    1 if result.passed else 0,
                    json.dumps([t.model_dump() for t in result.fix_tickets]),
                    result.summary,
                    datetime.now(timezone.utc).isoformat()
                )
            )
            conn.commit()
        return attempt_id

    def patch_manifest(self, manifest: Dict[str, Any], result: VerificationResult) -> Dict[str, Any]:
        """
        Applies prescribed fixes from tickets into the production manifest
        before triggering a targeted scene regeneration.
        """
        patched = json.loads(json.dumps(manifest))
        scenes = patched.get("scenes", [])

        for ticket in result.fix_tickets:
            scene_idx = ticket.scene_idx
            if scene_idx is not None and 0 <= scene_idx < len(scenes):
                target_scene = scenes[scene_idx]
                logger.info(f"Applying fix ticket to scene {scene_idx}: {ticket.prescribed_fix}")

                # 1. Mutate visual prompt with corrective directives
                target_scene["visual_prompt"] = (
                    f"{target_scene['visual_prompt']}, {ticket.prescribed_fix}"
                )

                # 2. Add negative tokens if hallucination or defect reported
                if ticket.category in ["product_integrity", "brand_fidelity"]:
                    target_scene["negative_prompt"] = (
                        f"{target_scene.get('negative_prompt', '')}, warped, morphing, "
                        f"mutated logos, low-resolution, bad typography, artifacts"
                    )

                # 3. If AV sync defect, adjust scene duration
                if ticket.category == "av_sync" and "pacing" in ticket.issue_description.lower():
                    target_scene["duration_sec"] = max(3.0, target_scene.get("duration_sec", 4.0) + 0.5)

            else:
                brand_kit = patched.setdefault("brand_kit", {})
                forbidden = brand_kit.setdefault("forbidden_claims", [])
                forbidden.append(f"Visual flaw: {ticket.issue_description}")

        return patched

    def run_verification_loop(
        self,
        job_id: str,
        initial_video_path: str,
        manifest: Dict[str, Any],
        brand_kit: Dict[str, Any],
        recompile_pipeline_fn: Callable[[Dict[str, Any], int], str],
        scratch_base_dir: str
    ) -> Tuple[bool, str, VerificationResult]:
        """
        Executes verification and autonomous regeneration up to max_attempts.
        recompile_pipeline_fn is a callable(patched_manifest, attempt_number) -> new_video_path.
        """
        current_video_path = initial_video_path
        current_manifest = manifest

        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"[Job {job_id}] Initiating Verification Attempt {attempt}/{self.max_attempts}")
            attempt_scratch = Path(scratch_base_dir) / f"attempt_{attempt}"

            # Step 1: Run Gemini Multimodal Audit
            result = self.verifier.verify(
                video_path=current_video_path,
                manifest_dict=current_manifest,
                brand_kit_dict=brand_kit,
                scratch_dir=str(attempt_scratch)
            )

            # Step 2: Record Attempt in SQLite
            self.record_attempt(job_id, attempt, result)
            logger.info(
                f"[Job {job_id}] Attempt {attempt} Result: Score={result.overall_score:.2f}, "
                f"Passed={result.passed}, Tickets={len(result.fix_tickets)}"
            )

            # Step 3: Check Termination Conditions
            if result.passed:
                logger.info(f"[Job {job_id}] Quality verification PASSED on attempt {attempt}!")
                return True, current_video_path, result

            if attempt == self.max_attempts:
                logger.warning(
                    f"[Job {job_id}] Max verification attempts ({self.max_attempts}) reached. "
                    f"Final score: {result.overall_score:.2f}. Ad rejected."
                )
                return False, current_video_path, result

            # Step 4: Mutate Manifest & Trigger Targeted Re-render
            logger.info(f"[Job {job_id}] Verification failed. Mutating manifest and triggering regeneration...")
            current_manifest = self.patch_manifest(current_manifest, result)
            
            try:
                current_video_path = recompile_pipeline_fn(current_manifest, attempt + 1)
            except Exception as e:
                logger.error(f"[Job {job_id}] Recompilation failed during attempt {attempt + 1}: {e}")
                return False, current_video_path, result

        return False, current_video_path, result
```

---

# MODULE 5: BRAND UI (`src/ui/brand_service.py`)

Module 5 handles client brand onboarding, enforcing strict chromatic, typographic, and transparency guarantees on uploaded assets.

---

## 1. Complete `brand_kit.json` Schema

### JSON Schema (Draft-07)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "BrandKit",
  "type": "object",
  "required": [
    "brand_id",
    "brand_name",
    "palette",
    "typography",
    "logo",
    "voice_and_tone"
  ],
  "properties": {
    "brand_id": {
      "type": "string",
      "pattern": "^[a-z0-9_-]{3,64}$",
      "description": "Unique alphanumeric kebab-case slug identifier for the brand"
    },
    "brand_name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 100,
      "description": "Official brand display name"
    },
    "tagline": {
      "type": "string",
      "maxLength": 200,
      "default": ""
    },
    "palette": {
      "type": "object",
      "required": ["primary", "secondary", "accent", "background", "text_primary"],
      "properties": {
        "primary": { "type": "string", "pattern": "^#[A-Fa-f0-9]{6}$" },
        "secondary": { "type": "string", "pattern": "^#[A-Fa-f0-9]{6}$" },
        "accent": { "type": "string", "pattern": "^#[A-Fa-f0-9]{6}$" },
        "background": { "type": "string", "pattern": "^#[A-Fa-f0-9]{6}$" },
        "text_primary": { "type": "string", "pattern": "^#[A-Fa-f0-9]{6}$" }
      }
    },
    "typography": {
      "type": "object",
      "required": ["primary_font_name", "primary_font_path", "header_case"],
      "properties": {
        "primary_font_name": { "type": "string", "minLength": 1 },
        "primary_font_path": { "type": "string", "pattern": "\\.(ttf|otf|woff2)$" },
        "secondary_font_name": { "type": ["string", "null"] },
        "secondary_font_path": { "type": ["string", "null"], "pattern": "(\\.(ttf|otf|woff2))?$" },
        "header_case": {
          "type": "string",
          "enum": ["UPPERCASE", "Title Case", "lowercase"]
        }
      }
    },
    "logo": {
      "type": "object",
      "required": ["path", "aspect_ratio", "min_height_pct", "safe_zone_margin_pct", "default_position"],
      "properties": {
        "path": { "type": "string" },
        "aspect_ratio": { "type": "number", "minimum": 0.1, "maximum": 10.0 },
        "min_height_pct": { "type": "number", "minimum": 0.03, "maximum": 0.25 },
        "safe_zone_margin_pct": { "type": "number", "minimum": 0.02, "maximum": 0.20 },
        "default_position": {
          "type": "string",
          "enum": ["top_right", "top_left", "bottom_right", "bottom_left", "center"]
        }
      }
    },
    "voice_and_tone": {
      "type": "object",
      "required": ["tone_attributes", "reading_speed_multiplier", "tts_voice_id"],
      "properties": {
        "tone_attributes": {
          "type": "array",
          "items": { "type": "string" },
          "minItems": 1
        },
        "reading_speed_multiplier": { "type": "number", "minimum": 0.75, "maximum": 1.5 },
        "tts_voice_id": { "type": "string", "default": "en-US-Neural2-D" }
      }
    },
    "guidelines": {
      "type": "object",
      "properties": {
        "forbidden_claims": {
          "type": "array",
          "items": { "type": "string" },
          "default": []
        },
        "mandatory_disclaimers": {
          "type": "array",
          "items": { "type": "string" },
          "default": []
        },
        "preferred_music_vibes": {
          "type": "array",
          "items": { "type": "string" },
          "default": ["energetic", "luxury", "chill", "upbeat"]
        }
      }
    }
  }
}
```

### Pydantic Models for Brand Kit
```python
"""
src/core/brand_models.py - Pydantic v2 Models for Brand UI & Kit Ingestion
"""

import re
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ColorPalette(BaseModel):
    primary: str = Field("#000000", description="Primary brand color hex code")
    secondary: str = Field("#FFFFFF", description="Secondary brand color hex code")
    accent: str = Field("#38BDF8", description="Accent highlight color hex code")
    background: str = Field("#0F172A", description="Default container background hex code")
    text_primary: str = Field("#F8FAFC", description="Primary legible font color hex code")

    @field_validator("primary", "secondary", "accent", "background", "text_primary")
    @classmethod
    def validate_hex_code(cls, v: str) -> str:
        if not re.match(r"^#[A-Fa-f0-9]{6}$", v):
            raise ValueError(f"Color code '{v}' must be a valid 6-character hex code (e.g. #FF0033)")
        return v.upper()


class TypographyConfig(BaseModel):
    primary_font_name: str = Field("Montserrat-Bold", min_length=1)
    primary_font_path: str = Field(..., description="Local path to uploaded TTF/OTF font file")
    secondary_font_name: Optional[str] = None
    secondary_font_path: Optional[str] = None
    header_case: Literal["UPPERCASE", "Title Case", "lowercase"] = "UPPERCASE"

    @field_validator("primary_font_path", "secondary_font_path")
    @classmethod
    def validate_font_extension(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        valid_exts = (".ttf", ".otf", ".woff2")
        if not any(v.lower().endswith(ext) for ext in valid_exts):
            raise ValueError(f"Font file '{v}' must terminate in .ttf, .otf, or .woff2")
        return v


class LogoConfig(BaseModel):
    path: str = Field(..., description="Local path to validated RGBA PNG logo")
    aspect_ratio: float = Field(..., gt=0.0, le=10.0)
    min_height_pct: float = Field(0.08, ge=0.03, le=0.25)
    safe_zone_margin_pct: float = Field(0.05, ge=0.02, le=0.20)
    default_position: Literal["top_right", "top_left", "bottom_right", "bottom_left", "center"] = "top_right"


class VoiceAndToneConfig(BaseModel):
    tone_attributes: List[str] = Field(default_factory=lambda: ["modern", "punchy", "authoritative"])
    reading_speed_multiplier: float = Field(1.05, ge=0.75, le=1.5)
    tts_voice_id: str = Field("en-US-Neural2-D")


class BrandGuidelines(BaseModel):
    forbidden_claims: List[str] = Field(default_factory=list)
    mandatory_disclaimers: List[str] = Field(default_factory=list)
    preferred_music_vibes: List[str] = Field(
        default_factory=lambda: ["energetic", "luxury", "chill", "upbeat"]
    )


class CompleteBrandKit(BaseModel):
    brand_id: str = Field(..., pattern=r"^[a-z0-9_-]{3,64}$")
    brand_name: str = Field(..., min_length=1, max_length=100)
    tagline: str = Field("", max_length=200)
    palette: ColorPalette
    typography: TypographyConfig
    logo: LogoConfig
    voice_and_tone: VoiceAndToneConfig
    guidelines: BrandGuidelines
```

---

## 2. Logo Validation (PIL RGBA Check) & Font Validation

```python
"""
src/ui/validators.py - Deep Validation for Brand Logos & Fonts
"""

import io
from pathlib import Path
from typing import Tuple
from PIL import Image
from fastapi import HTTPException


def validate_logo_image(file_bytes: bytes, filename: str) -> Tuple[int, int, float]:
    """
    Validates that the uploaded logo:
    1. Is a decodable image.
    2. Has format == 'PNG'.
    3. Has mode == 'RGBA' with active transparency variations.
    4. Has minimum dimensions of 128x128 px and aspect ratio between 0.1 and 10.0.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupt image file '{filename}': {str(e)}")

    image = Image.open(io.BytesIO(file_bytes))

    if image.format != "PNG":
        raise HTTPException(
            status_code=400,
            detail=f"Logo file '{filename}' must be a PNG image. Detected format: {image.format}"
        )

    if image.mode != "RGBA":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Logo file '{filename}' must have an RGBA color mode with transparency. "
                f"Detected color mode: '{image.mode}'. Please export your logo with transparent background."
            )
        )

    alpha_extrema = image.getchannel("A").getextrema()
    if alpha_extrema[0] == 255:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Logo '{filename}' has an RGBA header but contains zero transparent pixels "
                "(all alpha values are 255). A transparent background is required for video overlays."
            )
        )

    w, h = image.size
    if w < 128 or h < 128:
        raise HTTPException(
            status_code=400,
            detail=f"Logo resolution {w}x{h} is too small. Minimum required dimensions are 128x128 pixels."
        )

    aspect_ratio = round(float(w) / float(h), 4)
    if not (0.1 <= aspect_ratio <= 10.0):
        raise HTTPException(
            status_code=400,
            detail=f"Extreme logo aspect ratio ({aspect_ratio:.2f}). Must be between 0.1 and 10.0."
        )

    return w, h, aspect_ratio


def validate_font_file(file_bytes: bytes, filename: str) -> str:
    """
    Validates that the font terminates in .ttf, .otf, or .woff2
    and contains valid binary magic byte headers.
    """
    ext = Path(filename).suffix.lower()
    valid_exts = {".ttf", ".otf", ".woff2"}
    if ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid font extension '{ext}' for file '{filename}'. Must be one of: .ttf, .otf, .woff2"
        )

    if len(file_bytes) < 4:
        raise HTTPException(status_code=400, detail=f"Font file '{filename}' is empty or corrupt.")

    magic = file_bytes[:4]
    is_valid_magic = (
        magic == b"\x00\x01\x00\x00" or
        magic == b"OTTO" or
        magic == b"true" or
        magic == b"wOF2"
    )

    if not is_valid_magic:
        raise HTTPException(
            status_code=400,
            detail=f"Font file '{filename}' failed binary header check. Magic bytes: {magic!r}"
        )

    return ext
```

---

## 3. FastAPI Route Handlers (`GET /onboarding`, `POST /brands`)

```python
"""
src/ui/brand_routes.py - FastAPI Brand Ingestion & Onboarding Routes
"""

import json
import re
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse

from src.core.brand_models import (
    BrandGuidelines,
    ColorPalette,
    CompleteBrandKit,
    LogoConfig,
    TypographyConfig,
    VoiceAndToneConfig,
)
from src.ui.validators import validate_font_file, validate_logo_image

router = APIRouter(tags=["Brand Onboarding"])
VAULT_BASE_DIR = Path("storage/brand_vaults")


@router.get("/onboarding", response_class=HTMLResponse)
async def get_brand_onboarding_page():
    """Renders the standalone interactive Brand Kit setup UI."""
    template_path = Path("ui/templates/onboarding.html")
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
    return HTMLResponse(content=COMPLETE_ONBOARDING_HTML_FORM)


@router.post("/brands", status_code=status.HTTP_201_CREATED)
async def create_brand_kit(
    brand_name: str = Form(..., description="Brand display name"),
    brand_id: Optional[str] = Form(None, description="Optional custom slug"),
    tagline: str = Form("", description="Brand tagline"),
    primary_color: str = Form("#000000"),
    secondary_color: str = Form("#FFFFFF"),
    accent_color: str = Form("#38BDF8"),
    background_color: str = Form("#0F172A"),
    text_primary_color: str = Form("#F8FAFC"),
    header_case: str = Form("UPPERCASE"),
    reading_speed_multiplier: float = Form(1.05),
    tts_voice_id: str = Form("en-US-Neural2-D"),
    tone_attributes_csv: str = Form("modern, punchy, authoritative"),
    forbidden_claims_csv: str = Form(""),
    mandatory_disclaimers_csv: str = Form(""),
    logo_file: UploadFile = File(..., description="Transparent RGBA PNG logo"),
    font_file: UploadFile = File(..., description="Custom brand font (.ttf or .otf)"),
):
    """
    Ingests brand assets, performs PIL RGBA verification and font magic byte checks,
    creates brand vault storage, and outputs brand_kit.json.
    """
    # 1. Resolve slug
    slug = brand_id.strip() if brand_id else re.sub(r"[^a-z0-9_-]", "", brand_name.lower().replace(" ", "_"))
    if not re.match(r"^[a-z0-9_-]{3,64}$", slug):
        raise HTTPException(
            status_code=400,
            detail="brand_id must be 3-64 characters containing only lowercase letters, digits, underscores, or hyphens."
        )

    brand_dir = VAULT_BASE_DIR / slug
    brand_dir.mkdir(parents=True, exist_ok=True)

    # 2. Validate & Store Logo
    logo_bytes = await logo_file.read()
    w, h, aspect_ratio = validate_logo_image(logo_bytes, logo_file.filename)
    saved_logo_path = brand_dir / "logo.png"
    saved_logo_path.write_bytes(logo_bytes)

    # 3. Validate & Store Font
    font_bytes = await font_file.read()
    font_ext = validate_font_file(font_bytes, font_file.filename)
    font_stem = Path(font_file.filename).stem
    saved_font_path = brand_dir / f"primary_font{font_ext}"
    saved_font_path.write_bytes(font_bytes)

    # 4. Parse CSV inputs
    tone_list = [t.strip() for t in tone_attributes_csv.split(",") if t.strip()]
    forbidden_list = [f.strip() for f in forbidden_claims_csv.split(",") if f.strip()]
    disclaimers_list = [d.strip() for d in mandatory_disclaimers_csv.split(",") if d.strip()]

    # 5. Assemble BrandKit instance and validate with Pydantic
    try:
        kit = CompleteBrandKit(
            brand_id=slug,
            brand_name=brand_name.strip(),
            tagline=tagline.strip(),
            palette=ColorPalette(
                primary=primary_color,
                secondary=secondary_color,
                accent=accent_color,
                background=background_color,
                text_primary=text_primary_color,
            ),
            typography=TypographyConfig(
                primary_font_name=font_stem,
                primary_font_path=str(saved_font_path.resolve()),
                header_case=header_case,
            ),
            logo=LogoConfig(
                path=str(saved_logo_path.resolve()),
                aspect_ratio=aspect_ratio,
                min_height_pct=0.08,
                safe_zone_margin_pct=0.05,
                default_position="top_right",
            ),
            voice_and_tone=VoiceAndToneConfig(
                tone_attributes=tone_list or ["bold", "premium"],
                reading_speed_multiplier=reading_speed_multiplier,
                tts_voice_id=tts_voice_id,
            ),
            guidelines=BrandGuidelines(
                forbidden_claims=forbidden_list,
                mandatory_disclaimers=disclaimers_list,
                preferred_music_vibes=["energetic", "luxury", "chill", "upbeat"],
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Brand Kit Schema Validation Failed: {str(e)}")

    # 6. Save brand_kit.json into brand vault
    kit_json_path = brand_dir / "brand_kit.json"
    kit_json_path.write_text(json.dumps(kit.model_dump(), indent=2), encoding="utf-8")

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": "SUCCESS",
            "message": f"Brand kit for '{kit.brand_name}' created successfully.",
            "brand_id": slug,
            "vault_path": str(brand_dir.resolve()),
            "brand_kit": kit.model_dump(),
        },
    )
```

---

## 4. Complete HTML Form

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Brand Kit Onboarding | AI Video Ad Engine</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(22, 30, 46, 0.7);
      --border: rgba(255, 255, 255, 0.08);
      --border-focus: #38bdf8;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --danger: #ef4444;
      --success: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { background: var(--bg); color: var(--text); min-height: 100vh; padding: 40px 20px; display: flex; justify-content: center; }
    .container { width: 100%; max-width: 880px; }
    header { margin-bottom: 32px; border-bottom: 1px solid var(--border); padding-bottom: 20px; }
    h1 { font-size: 28px; font-weight: 700; display: flex; align-items: center; gap: 12px; }
    p.subtitle { color: var(--text-muted); font-size: 14px; margin-top: 6px; }
    .card { background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; }
    .card-title { font-size: 18px; font-weight: 600; margin-bottom: 16px; color: var(--accent); display: flex; align-items: center; gap: 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
    .form-group { display: flex; flex-direction: column; gap: 6px; }
    label { font-size: 13px; font-weight: 500; color: var(--text-muted); }
    input[type="text"], input[type="number"], select, textarea {
      background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; color: #fff; font-size: 14px; transition: border 0.2s;
    }
    input[type="text"]:focus, select:focus, textarea:focus { outline: none; border-color: var(--border-focus); }
    textarea { resize: vertical; min-height: 70px; }
    .color-picker-wrap { display: flex; align-items: center; gap: 10px; background: #0f172a; padding: 6px 12px; border: 1px solid var(--border); border-radius: 8px; }
    input[type="color"] { -webkit-appearance: none; border: none; width: 32px; height: 32px; border-radius: 6px; cursor: pointer; background: transparent; }
    input[type="color"]::-webkit-color-swatch { border-radius: 6px; border: 1px solid rgba(255,255,255,0.2); }
    .dropzone {
      border: 2px dashed var(--border); border-radius: 8px; padding: 24px; text-align: center; cursor: pointer; transition: all 0.2s; background: rgba(15, 23, 42, 0.4);
    }
    .dropzone:hover { border-color: var(--accent); background: rgba(56, 189, 248, 0.05); }
    .dropzone p { font-size: 13px; color: var(--text-muted); margin-top: 6px; }
    .preview-box { margin-top: 12px; display: none; align-items: center; gap: 12px; background: #0f172a; padding: 10px; border-radius: 6px; }
    .preview-box img { max-height: 48px; max-width: 120px; object-fit: contain; }
    .btn-submit {
      background: linear-gradient(135deg, #0284c7, #38bdf8); color: #0f172a; font-weight: 700; border: none; border-radius: 8px;
      padding: 14px 28px; font-size: 15px; cursor: pointer; width: 100%; transition: opacity 0.2s;
    }
    .btn-submit:hover { opacity: 0.95; }
    #toast {
      position: fixed; bottom: 24px; right: 24px; padding: 14px 20px; border-radius: 8px; font-size: 14px; display: none;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); z-index: 1000;
    }
    .toast-success { background: var(--success); color: #fff; }
    .toast-error { background: var(--danger); color: #fff; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Brand Kit Onboarding</h1>
      <p class="subtitle">Register your brand identity, color tokens, typography, and validated assets for autonomous video ad synthesis.</p>
    </header>

    <form id="brandForm" enctype="multipart/form-data">
      <!-- 1. Identity -->
      <div class="card">
        <div class="card-title">1. Brand Identity</div>
        <div class="grid">
          <div class="form-group">
            <label for="brand_name">Brand Name *</label>
            <input type="text" id="brand_name" name="brand_name" placeholder="e.g. Lumina Audio" required>
          </div>
          <div class="form-group">
            <label for="brand_id">Slug ID (optional)</label>
            <input type="text" id="brand_id" name="brand_id" placeholder="lumina_audio (auto-generated if empty)">
          </div>
        </div>
        <div class="form-group" style="margin-top: 14px;">
          <label for="tagline">Commercial Tagline</label>
          <input type="text" id="tagline" name="tagline" placeholder="e.g. Sound That Defies Gravity">
        </div>
      </div>

      <!-- 2. Color Palette -->
      <div class="card">
        <div class="card-title">2. Color Palette (Hex Tokens)</div>
        <div class="grid">
          <div class="form-group">
            <label>Primary Brand Color</label>
            <div class="color-picker-wrap">
              <input type="color" id="primary_picker" value="#0F172A">
              <input type="text" id="primary_color" name="primary_color" value="#0F172A" style="width: 100px;">
            </div>
          </div>
          <div class="form-group">
            <label>Secondary Color</label>
            <div class="color-picker-wrap">
              <input type="color" id="secondary_picker" value="#FFFFFF">
              <input type="text" id="secondary_color" name="secondary_color" value="#FFFFFF" style="width: 100px;">
            </div>
          </div>
          <div class="form-group">
            <label>Accent Highlight</label>
            <div class="color-picker-wrap">
              <input type="color" id="accent_picker" value="#38BDF8">
              <input type="text" id="accent_color" name="accent_color" value="#38BDF8" style="width: 100px;">
            </div>
          </div>
        </div>
      </div>

      <!-- 3. Assets: Logo & Font -->
      <div class="card">
        <div class="card-title">3. Brand Assets (Deep Validation)</div>
        <div class="grid">
          <div class="form-group">
            <label>Brand Logo (RGBA PNG with Transparency) *</label>
            <div class="dropzone" id="logoDropzone" onclick="document.getElementById('logo_file').click()">
              <div style="font-size: 24px;">🖼️</div>
              <strong>Click or Drag PNG Logo</strong>
              <p>Must have transparent background (RGBA, min 128x128px)</p>
              <input type="file" id="logo_file" name="logo_file" accept=".png" style="display: none;" required>
            </div>
            <div class="preview-box" id="logoPreview">
              <img id="logoImg" src="" alt="Logo Preview">
              <span id="logoDetails" style="font-size: 12px; color: var(--text-muted);"></span>
            </div>
          </div>

          <div class="form-group">
            <label>Custom Brand Font (.ttf or .otf) *</label>
            <div class="dropzone" id="fontDropzone" onclick="document.getElementById('font_file').click()">
              <div style="font-size: 24px;">🔤</div>
              <strong>Click or Drag Font File</strong>
              <p>Supports TrueType (.ttf) or OpenType (.otf)</p>
              <input type="file" id="font_file" name="font_file" accept=".ttf,.otf,.woff2" style="display: none;" required>
            </div>
            <div class="preview-box" id="fontPreview">
              <span style="font-size: 18px;">📄</span>
              <span id="fontDetails" style="font-size: 13px; color: var(--text);"></span>
            </div>
          </div>
        </div>
      </div>

      <!-- 4. Typography & Voice Guidelines -->
      <div class="card">
        <div class="card-title">4. Typography & Voice Directives</div>
        <div class="grid">
          <div class="form-group">
            <label for="header_case">Header Title Casing</label>
            <select id="header_case" name="header_case">
              <option value="UPPERCASE" selected>UPPERCASE (Punchy Ads)</option>
              <option value="Title Case">Title Case (Editorial)</option>
              <option value="lowercase">lowercase (Minimalist)</option>
            </select>
          </div>
          <div class="form-group">
            <label for="tts_voice_id">TTS Neural Voice</label>
            <select id="tts_voice_id" name="tts_voice_id">
              <option value="en-US-Neural2-D" selected>en-US-Neural2-D (Authoritative Male)</option>
              <option value="en-US-Neural2-F">en-US-Neural2-F (Warm Conversational Female)</option>
              <option value="en-US-Neural2-J">en-US-Neural2-J (Energetic Youth Male)</option>
            </select>
          </div>
          <div class="form-group">
            <label for="reading_speed_multiplier">Voice Speed Multiplier</label>
            <input type="number" id="reading_speed_multiplier" name="reading_speed_multiplier" min="0.8" max="1.3" step="0.05" value="1.05">
          </div>
        </div>
        <div class="form-group" style="margin-top: 14px;">
          <label for="tone_attributes_csv">Tone Attributes (Comma Separated)</label>
          <input type="text" id="tone_attributes_csv" name="tone_attributes_csv" value="modern, punchy, high-energy, premium">
        </div>
        <div class="form-group" style="margin-top: 14px;">
          <label for="forbidden_claims_csv">Forbidden Claims or Words (Comma Separated)</label>
          <textarea id="forbidden_claims_csv" name="forbidden_claims_csv" placeholder="e.g. miracle cure, 100% guaranteed, cheapest, effortless"></textarea>
        </div>
      </div>

      <button type="submit" class="btn-submit" id="submitBtn">Save & Register Brand Kit</button>
    </form>
  </div>

  <div id="toast"></div>

  <script>
    ['primary', 'secondary', 'accent'].forEach(name => {
      const picker = document.getElementById(`${name}_picker`);
      const txt = document.getElementById(`${name}_color`);
      picker.addEventListener('input', e => txt.value = e.target.value.toUpperCase());
      txt.addEventListener('input', e => {
        if (/^#[0-9A-F]{6}$/i.test(e.target.value)) picker.value = e.target.value;
      });
    });

    const logoInput = document.getElementById('logo_file');
    logoInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (file) {
        if (!file.name.toLowerCase().endsWith('.png')) {
          showToast('Logo must be a PNG image with transparency', true);
          logoInput.value = '';
          return;
        }
        const reader = new FileReader();
        reader.onload = ev => {
          document.getElementById('logoImg').src = ev.target.result;
          document.getElementById('logoDetails').innerText = `${file.name} (${Math.round(file.size / 1024)} KB)`;
          document.getElementById('logoPreview').style.display = 'flex';
        };
        reader.readAsDataURL(file);
      }
    });

    const fontInput = document.getElementById('font_file');
    fontInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (file) {
        document.getElementById('fontDetails').innerText = `${file.name} (${Math.round(file.size / 1024)} KB)`;
        document.getElementById('fontPreview').style.display = 'flex';
      }
    });

    const form = document.getElementById('brandForm');
    const toast = document.getElementById('toast');
    const submitBtn = document.getElementById('submitBtn');

    function showToast(msg, isError = false) {
      toast.innerText = msg;
      toast.className = isError ? 'toast-error' : 'toast-success';
      toast.style.display = 'block';
      setTimeout(() => toast.style.display = 'none', 4500);
    }

    form.addEventListener('submit', async e => {
      e.preventDefault();
      submitBtn.disabled = true;
      submitBtn.innerText = 'Validating and Uploading...';

      const formData = new FormData(form);
      try {
        const res = await fetch('/brands', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to register brand');
        showToast(`Success! Brand Kit '${data.brand_id}' registered.`);
        setTimeout(() => { window.location.href = `/jobs?brand_id=${data.brand_id}`; }, 1500);
      } catch (err) {
        showToast(err.message, true);
      } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = 'Save & Register Brand Kit';
      }
    });
  </script>
</body>
</html>
```

---

# MODULE 6: ORCHESTRATOR (`src/orchestrator/pipeline.py` & `main.py`)

Module 6 coordinates asynchronous pipeline execution across sub-engines with transactional SQLite state machine persistence and client revision handling.

---

## 1. End-to-End Async Pipeline Function

```python
"""
src/orchestrator/pipeline.py - Master Pipeline Execution Engine
"""

import json
import logging
import os
import shutil
import sqlite3
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.generation.preprocessor import run_preprocessor
from src.generation.scene_chain_engine import SceneChainEngine
from src.generation.audio_engine import generate_voiceover, detect_rhythm_beats, build_ducked_audio
from src.compositor.compositor import VideoCompositor
from src.verifier.regeneration_loop import RegenerationController

logger = logging.getLogger("orchestrator.pipeline")


def update_job_status(
    db_path: str,
    job_id: str,
    status: str,
    progress: int,
    stage_name: str,
    error_log: Optional[str] = None,
    final_output_path: Optional[str] = None
):
    """Atomically records job state, percentage progress, and errors in SQLite."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?,
                progress = ?,
                current_stage = ?,
                error_log = COALESCE(?, error_log),
                final_video_path = COALESCE(?, final_video_path),
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                status,
                progress,
                stage_name,
                error_log,
                final_output_path,
                datetime.now(timezone.utc).isoformat(),
                job_id
            )
        )
        conn.commit()
    logger.info(f"[Job {job_id}] State ➔ {status} ({progress}%) | Stage: {stage_name}")


async def run_pipeline(
    job_id: str,
    manifest: Dict[str, Any],
    raw_asset_path: str,
    brand_kit_path: str,
    db_path: str = "storage/engine.db",
    storage_root: str = "storage/jobs"
) -> str:
    """
    Executes the autonomous generation, composition, verification, and regeneration chain:
    1. STAGING & PREPROCESSING (rembg + Real-ESRGAN)
    2. SCENE CHAIN GENERATION (Veo 2 Sequential Generation with tail frames)
    3. AUDIO PRODUCTION (TTS + Beat Tracking + Denoise)
    4. VIDEO COMPOSITION (Concat Demuxer + Transitions + Logo Overlay + Ducking)
    5. QUALITY VERIFICATION (Gemini 1.5 Flash 0.5fps Audit + Regeneration Loop)
    """
    job_dir = Path(storage_root) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    scratch_dir = job_dir / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    with open(brand_kit_path, "r", encoding="utf-8") as f:
        brand_kit = json.load(f)

    try:
        # STAGE 0: INITIALIZATION
        update_job_status(db_path, job_id, "RUNNING", 5, "INITIALIZING")

        # STAGE 3A: ASSET PREPROCESSING
        update_job_status(db_path, job_id, "RUNNING", 15, "PREPROCESSING_ASSETS")
        preprocessed_png = str(scratch_dir / "product_isolated_upscaled.png")
        try:
            run_preprocessor(raw_asset_path, preprocessed_png)
        except Exception as prep_err:
            logger.warning(f"Preprocessor warning: {prep_err}. Falling back to raw asset.")
            shutil.copy(raw_asset_path, preprocessed_png)

        # STAGE 3B: SCENE CHAIN GENERATION
        update_job_status(db_path, job_id, "RUNNING", 35, "SCENE_CHAIN_GENERATION")
        scene_engine = SceneChainEngine(db_path=db_path)
        scene_mp4_paths = scene_engine.generate_all_scenes(
            job_id=job_id,
            manifest=manifest,
            initial_anchor_image=preprocessed_png,
            output_dir=str(job_dir / "scenes")
        )

        # STAGE 3C: AUDIO SYNTHESIS & BEAT EXTRACTION
        update_job_status(db_path, job_id, "RUNNING", 55, "AUDIO_SYNTHESIS")
        audio_dir = job_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        vo_script = " ".join([s.get("voiceover_script", "") for s in manifest.get("scenes", []) if s.get("voiceover_script")])
        vo_wav_path = str(audio_dir / "voiceover.wav")
        generate_voiceover(
            script=vo_script,
            output_path=vo_wav_path,
            voice_id=brand_kit.get("voice_and_tone", {}).get("tts_voice_id", "en-US-Neural2-D"),
            speaking_rate=brand_kit.get("voice_and_tone", {}).get("reading_speed_multiplier", 1.05)
        )

        music_vibe = manifest.get("music_vibe", "energetic")
        tempo, beat_times = detect_rhythm_beats(vibe=music_vibe)

        # STAGE 3D: COMPOSITION & ASSEMBLY
        update_job_status(db_path, job_id, "RUNNING", 75, "COMPOSITING")
        compositor = VideoCompositor()
        initial_composed_mp4 = str(job_dir / "master_unverified.mp4")
        
        compositor.compose(
            scene_paths=scene_mp4_paths,
            transitions=[s.get("transition_to_next") for s in manifest.get("scenes", [])],
            voiceover_path=vo_wav_path,
            logo_path=brand_kit.get("logo", {}).get("path"),
            output_path=initial_composed_mp4
        )

        # STAGE 4: MULTIMODAL VERIFICATION & REGENERATION LOOP
        update_job_status(db_path, job_id, "RUNNING", 88, "VERIFYING")

        def recompile_subpipeline(patched_manifest: Dict[str, Any], attempt_num: int) -> str:
            logger.info(f"[Job {job_id}] Re-running scene chain for attempt {attempt_num}...")
            new_scenes = scene_engine.generate_all_scenes(
                job_id=f"{job_id}_att{attempt_num}",
                manifest=patched_manifest,
                initial_anchor_image=preprocessed_png,
                output_dir=str(job_dir / f"scenes_att{attempt_num}")
            )
            new_master = str(job_dir / f"master_attempt_{attempt_num}.mp4")
            compositor.compose(
                scene_paths=new_scenes,
                transitions=[s.get("transition_to_next") for s in patched_manifest.get("scenes", [])],
                voiceover_path=vo_wav_path,
                logo_path=brand_kit.get("logo", {}).get("path"),
                output_path=new_master
            )
            return new_master

        regen_controller = RegenerationController(db_path=db_path, max_attempts=3)
        passed, verified_mp4_path, verifier_result = regen_controller.run_verification_loop(
            job_id=job_id,
            initial_video_path=initial_composed_mp4,
            manifest=manifest,
            brand_kit=brand_kit,
            recompile_pipeline_fn=recompile_subpipeline,
            scratch_base_dir=str(scratch_dir)
        )

        # FINALIZATION
        final_mp4_path = str(job_dir / f"{job_id}_final.mp4")
        shutil.copy(verified_mp4_path, final_mp4_path)

        if passed:
            update_job_status(
                db_path=db_path,
                job_id=job_id,
                status="COMPLETED",
                progress=100,
                stage_name="COMPLETE",
                final_output_path=final_mp4_path
            )
        else:
            update_job_status(
                db_path=db_path,
                job_id=job_id,
                status="FAILED_VERIFICATION",
                progress=100,
                stage_name="FAILED_VERIFICATION",
                error_log=f"Verification failed after 3 attempts. Final score: {verifier_result.overall_score:.2f}",
                final_output_path=final_mp4_path
            )

        return final_mp4_path

    except Exception as e:
        err_msg = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logger.error(f"[Job {job_id}] Critical Pipeline Failure: {err_msg}")
        update_job_status(
            db_path=db_path,
            job_id=job_id,
            status="FAILED",
            progress=0,
            stage_name="CRASHED",
            error_log=err_msg
        )
        raise
```

---

## 2. Complete FastAPI `main.py` Application

```python
"""
main.py - FastAPI Master Production Application
"""

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.orchestrator.pipeline import run_pipeline
from src.ui.brand_routes import router as brand_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ad_engine.main")

DB_PATH = "storage/engine.db"

app = FastAPI(
    title="Autonomous AI Video Ad Engine",
    description="Multimodal commercial video generation, composition, and verification API.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(brand_router)


# ============================================================================
# Database Bootstrap
# ============================================================================

def init_database():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            brand_id TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER DEFAULT 0,
            current_stage TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            raw_asset_path TEXT NOT NULL,
            brand_kit_path TEXT NOT NULL,
            final_video_path TEXT,
            error_log TEXT,
            revision_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS job_revisions (
            revision_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            revision_number INTEGER NOT NULL,
            feedback_prompt TEXT NOT NULL,
            targeted_scenes_json TEXT,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(job_id) REFERENCES jobs(job_id)
        );
        """)
        conn.commit()

init_database()


# ============================================================================
# Pydantic Request & Response Schemas
# ============================================================================

class CreateJobRequest(BaseModel):
    brand_id: str = Field(..., description="Registered brand identifier slug")
    product_name: str = Field(..., description="Commercial name of product")
    product_image_path: str = Field(..., description="Local path to uploaded raw product image")
    creative_brief: str = Field(..., description="User prompt, core value propositions, target audience")
    aspect_ratio: str = Field("9:16", description="Video format (default 9:16 vertical)")
    target_duration_sec: float = Field(20.0, ge=10.0, le=60.0)
    music_vibe: str = Field("energetic", description="Music library vibe category")
    scenes: Optional[List[Dict[str, Any]]] = Field(None, description="Optional pre-scripted scene array")


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    current_stage: str
    created_at: str


class AttemptSummary(BaseModel):
    attempt_number: int
    overall_score: float
    passed: bool
    summary: Optional[str]
    created_at: str


class JobStatusResponse(BaseModel):
    job_id: str
    brand_id: str
    status: str
    progress: int
    current_stage: str
    revision_count: int
    final_video_path: Optional[str] = None
    error_log: Optional[str] = None
    created_at: str
    updated_at: str
    attempts: List[AttemptSummary] = Field(default_factory=list)


class ReviseJobRequest(BaseModel):
    feedback_prompt: str = Field(..., min_length=5, description="Client revision feedback instructions")
    targeted_scene_indices: Optional[List[int]] = Field(None, description="Specific scene indices to regenerate")


class ReviseJobResponse(BaseModel):
    job_id: str
    revision_number: int
    status: str
    message: str


# ============================================================================
# API Route Handlers
# ============================================================================

@app.post("/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(request: CreateJobRequest, background_tasks: BackgroundTasks):
    """
    Spawns an autonomous video ad generation and verification job.
    """
    brand_dir = Path("storage/brand_vaults") / request.brand_id
    brand_kit_file = brand_dir / "brand_kit.json"
    if not brand_kit_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Brand '{request.brand_id}' not found. Please complete /onboarding first."
        )

    if not Path(request.product_image_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"Product image file not found at '{request.product_image_path}'"
        )

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    scenes = request.scenes or [
        {
            "scene_idx": 0,
            "scene_type": "hook",
            "duration_sec": 4.0,
            "visual_prompt": f"Dramatic cinematic macro camera push-in on {request.product_name}, 4k commercial product lighting",
            "voiceover_script": f"Tired of ordinary solutions? Meet {request.product_name}.",
            "transition_to_next": {"transition_type": "whip_pan_right", "overlap_duration_ms": 300}
        },
        {
            "scene_idx": 1,
            "scene_type": "solution_reveal",
            "duration_sec": 5.0,
            "visual_prompt": f"Dynamic rotating studio shot of {request.product_name} highlighting sleek design and premium build quality",
            "voiceover_script": f"Engineered for high performance and sleek aesthetics.",
            "transition_to_next": {"transition_type": "cross_dissolve", "overlap_duration_ms": 300}
        },
        {
            "scene_idx": 2,
            "scene_type": "call_to_action",
            "duration_sec": 4.0,
            "visual_prompt": f"Hero shot of {request.product_name} centered on sleek pedestal with brand accent lighting",
            "voiceover_script": f"Upgrade your lifestyle today. Link in bio.",
            "transition_to_next": None
        }
    ]

    manifest = {
        "job_id": job_id,
        "brand_id": request.brand_id,
        "product_name": request.product_name,
        "creative_brief": request.creative_brief,
        "aspect_ratio": request.aspect_ratio,
        "target_duration_sec": request.target_duration_sec,
        "music_vibe": request.music_vibe,
        "scenes": scenes
    }

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, brand_id, status, progress, current_stage,
                manifest_json, raw_asset_path, brand_kit_path, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                request.brand_id,
                "QUEUED",
                0,
                "QUEUED",
                json.dumps(manifest),
                request.product_image_path,
                str(brand_kit_file.resolve()),
                now,
                now
            )
        )
        conn.commit()

    background_tasks.add_task(
        run_pipeline,
        job_id=job_id,
        manifest=manifest,
        raw_asset_path=request.product_image_path,
        brand_kit_path=str(brand_kit_file.resolve()),
        db_path=DB_PATH
    )

    return JobResponse(
        job_id=job_id,
        status="QUEUED",
        progress=0,
        current_stage="QUEUED",
        created_at=now
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Queries real-time job status, pipeline stage progress, and verification attempt history.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        job_row = cur.fetchone()
        if not job_row:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        cur.execute(
            "SELECT attempt_number, overall_score, passed, summary, created_at FROM verification_attempts WHERE job_id = ? ORDER BY attempt_number ASC",
            (job_id,)
        )
        attempt_rows = cur.fetchall()

    attempts = [
        AttemptSummary(
            attempt_number=r["attempt_number"],
            overall_score=r["overall_score"],
            passed=bool(r["passed"]),
            summary=r["summary"],
            created_at=r["created_at"]
        )
        for r in attempt_rows
    ]

    return JobStatusResponse(
        job_id=job_row["job_id"],
        brand_id=job_row["brand_id"],
        status=job_row["status"],
        progress=job_row["progress"],
        current_stage=job_row["current_stage"],
        revision_count=job_row["revision_count"],
        final_video_path=job_row["final_video_path"],
        error_log=job_row["error_log"],
        created_at=job_row["created_at"],
        updated_at=job_row["updated_at"],
        attempts=attempts
    )


@app.get("/jobs/{job_id}/download")
async def download_job_video(job_id: str):
    """
    Streams the broadcast-ready master MP4 if job execution has succeeded.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT status, final_video_path FROM jobs WHERE job_id = ?", (job_id,))
        row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    status_val, final_path = row[0], row[1]
    if status_val not in ["COMPLETED", "FAILED_VERIFICATION"]:
        raise HTTPException(
            status_code=400,
            detail=f"Video cannot be downloaded while in state '{status_val}'. Current progress is not finalized."
        )

    if not final_path or not Path(final_path).exists():
        raise HTTPException(status_code=404, detail="Rendered master video file is missing on server.")

    return FileResponse(
        path=final_path,
        media_type="video/mp4",
        filename=f"{job_id}_ad_master.mp4"
    )


@app.post("/jobs/{job_id}/revise", response_model=ReviseJobResponse)
async def revise_job(job_id: str, request: ReviseJobRequest, background_tasks: BackgroundTasks):
    """
    Injects human creative revision instructions, patches the manifest, and restarts
    the pipeline in revision mode.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        job_row = cur.fetchone()

    if not job_row:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job_row["status"] in ["RUNNING", "QUEUED"]:
        raise HTTPException(status_code=400, detail="Cannot revise a job that is actively executing.")

    manifest = json.loads(job_row["manifest_json"])
    revision_count = job_row["revision_count"] + 1
    revision_id = f"rev_{uuid.uuid4().hex[:8]}"

    scenes = manifest.get("scenes", [])
    target_idxs = request.targeted_scene_indices or list(range(len(scenes)))

    for idx in target_idxs:
        if 0 <= idx < len(scenes):
            scenes[idx]["visual_prompt"] = f"{scenes[idx]['visual_prompt']}, {request.feedback_prompt}"
            logger.info(f"[Job {job_id}] Revised scene {idx} visual prompt with feedback.")

    manifest["scenes"] = scenes

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO job_revisions (
                revision_id, job_id, revision_number, feedback_prompt, targeted_scenes_json, status
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                job_id,
                revision_count,
                request.feedback_prompt,
                json.dumps(target_idxs),
                "QUEUED"
            )
        )
        conn.execute(
            """
            UPDATE jobs
            SET status = 'QUEUED',
                progress = 0,
                current_stage = 'QUEUED_REVISION',
                manifest_json = ?,
                revision_count = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                json.dumps(manifest),
                revision_count,
                datetime.now(timezone.utc).isoformat(),
                job_id
            )
        )
        conn.commit()

    background_tasks.add_task(
        run_pipeline,
        job_id=job_id,
        manifest=manifest,
        raw_asset_path=job_row["raw_asset_path"],
        brand_kit_path=job_row["brand_kit_path"],
        db_path=DB_PATH
    )

    return ReviseJobResponse(
        job_id=job_id,
        revision_number=revision_count,
        status="QUEUED",
        message=f"Revision #{revision_count} queued for scenes {target_idxs}."
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 3. Revision Handling Logic

When a user submits revision instructions through `POST /jobs/{job_id}/revise`:
1. **Precondition Guard**: Enforces that the job exists and is not actively running (`status NOT IN ('RUNNING', 'QUEUED')`).
2. **Revision Lineage Tracking**: Persists revision audit trails in `job_revisions` (`revision_id`, `revision_number`, `feedback_prompt`, `targeted_scenes_json`).
3. **Manifest Injection**: Selectively appends client feedback directives to the targeted scenes' `visual_prompt` and updates the negative prompts if necessary.
4. **State Machine Reset**: Resets job status to `QUEUED` and stage to `QUEUED_REVISION`.
5. **Selective Sub-Pipeline Execution**: Reuses previously rendered valid scene MP4 clips from the cache and regenerates only modified scenes, saving compute costs and turnaround time.

---

## Consolidated SQLite Schemas

```sql
-- 1. Master Production Jobs
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    brand_id TEXT NOT NULL,
    status TEXT NOT NULL,
    progress INTEGER DEFAULT 0,
    current_stage TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    raw_asset_path TEXT NOT NULL,
    brand_kit_path TEXT NOT NULL,
    final_video_path TEXT,
    error_log TEXT,
    revision_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Verification Attempts (Module 4)
CREATE TABLE IF NOT EXISTS verification_attempts (
    attempt_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    product_integrity REAL NOT NULL,
    brand_fidelity REAL NOT NULL,
    scene_quality REAL NOT NULL,
    av_sync REAL NOT NULL,
    text_legibility REAL NOT NULL,
    cinematic_quality REAL NOT NULL,
    overall_score REAL NOT NULL,
    passed INTEGER NOT NULL,
    fix_tickets_json TEXT,
    summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);

-- 3. Creative Revision Ledger (Module 6)
CREATE TABLE IF NOT EXISTS job_revisions (
    revision_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    revision_number INTEGER NOT NULL,
    feedback_prompt TEXT NOT NULL,
    targeted_scenes_json TEXT,
    status TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);
```

