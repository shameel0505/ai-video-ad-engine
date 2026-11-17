# AI Video Ad Engine: Modules 4, 5, and 6 Technical Specification

---

## Architecture Overview: Stages 4, 5, and 6

Modules 4, 5, and 6 complete the end-to-end autonomous video ad engine. They govern brand asset onboarding, multimodal verification with automated regeneration, and top-level pipeline orchestration:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        MODULE 5: BRAND UI                              │
│  - GET /onboarding (Interactive Dark-Mode Brand Setup Form)           │
│  - POST /brands (Multipart Ingestion: Logo RGBA & TTF Font Validation) │
│  - brand_kit.json Schema Generation & Local Vault Persistence         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ brand_kit.json
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      MODULE 6: ORCHESTRATOR                            │
│  - FastAPI Engine (POST /jobs, GET /jobs/{id}, GET download, revise)   │
│  - SQLite State Machine (QUEUED ➔ STAGING ➔ SCENES ➔ AUDIO ➔ RENDER)   │
│  - Asynchronous Stage Execution & Task Management                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Rendered Master MP4
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        MODULE 4: VERIFIER                              │
│  - FFmpeg Temporal Frame Sampler (fps=0.5, scale=540:960, cap=15)      │
│  - Gemini 1.5 Flash Multimodal Quality & Hallucination Audit           │
│  - 6-Category Weighted Score Formula (Pass Threshold >= 85.0)          │
│  - SQLite Attempt Tracking & Autonomous Regeneration Loop              │
└────────────────────────────────────────────────────────────────────────┘
```

---

# MODULE 4: VERIFIER (`src/verifier/verifier.py`)

## 1. FFmpeg Frame Sampling Protocol

To perform multimodal quality verification without overflowing LLM context token windows or introducing network latency bottlenecks, the verifier extracts low-resolution visual keyframes at a conservative temporal frequency.

### Exact Sampling Command
```bash
ffmpeg -i "{video_path}" -vf "fps=0.5,scale=540:960:force_original_aspect_ratio=decrease,pad=540:960:(ow-iw)/2:(oh-ih)/2" -frames:v 15 -q:v 2 "{output_dir}/frame_%03d.jpg" -y
```

### Parameter Rationale & Breakdown
- `-i "{video_path}"`: Input master MP4 path.
- `-vf "fps=0.5,scale=540:960..."`:
  - `fps=0.5`: Samples one keyframe every $2.0$ seconds. A standard $15$-second ad yields $7$ frames; a $30$-second ad yields $15$ frames.
  - `scale=540:960:force_original_aspect_ratio=decrease`: Rescales the frame to fit inside a $540 \times 960$ vertical bounding box while preserving the source aspect ratio.
  - `pad=540:960:(ow-iw)/2:(oh-ih)/2`: Centers and letterboxes/pillarboxes the frame with black padding if the source aspect ratio deviates from $9:16$.
- `-frames:v 15` (or `-vframes 15`): Hard ceiling capping extraction at exactly $15$ frames. This guarantees that video ads exceeding $30$ seconds never exceed the target image token budget ($15 \text{ images} \times 258 \text{ tokens/image} \approx 3,870 \text{ tokens}$).
- `-q:v 2`: High-quality JPEG extraction preventing compression artifacts that could fool the vision model into penalizing image clarity.
- `-y`: Unconditionally overwrites existing frames in the destination directory.

---

## 2. Gemini 1.5 Flash Batch Multimodal Evaluator

The evaluator batches the extracted PIL images into a single multimodal request sent to `gemini-1.5-flash` using Google GenAI SDK, enforcing strict JSON output compliance.

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
    scene_idx: Optional[int] = Field(None, description="0-indexed scene number responsible")
    timestamp_sec: Optional[float] = Field(None, description="Timestamp in seconds")
    severity: str = Field(..., description="CRITICAL, MAJOR, or MINOR")
    issue_description: str = Field(..., description="Concise diagnosis of visual flaw")
    prescribed_fix: str = Field(..., description="Actionable prompt modification or parameter override")


class VerificationResult(BaseModel):
    product_integrity: float = Field(..., ge=0.0, le=1.0, description="Product visual fidelity (0.0-1.0)")
    brand_fidelity: float = Field(..., ge=0.0, le=1.0, description="Brand colors, typography, logos (0.0-1.0)")
    scene_quality: float = Field(..., ge=0.0, le=1.0, description="Coherence, temporal consistency (0.0-1.0)")
    av_sync: float = Field(..., ge=0.0, le=1.0, description="Audio-visual pacing and voice alignment (0.0-1.0)")
    text_legibility: float = Field(..., ge=0.0, le=1.0, description="OCR readability and safe-zone compliance (0.0-1.0)")
    cinematic_quality: float = Field(..., ge=0.0, le=1.0, description="Lighting, camera motion, composition (0.0-1.0)")
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Weighted composite score out of 100")
    passed: bool = Field(..., description="True if overall_score >= 85.0 and no CRITICAL tickets exist")
    summary: str = Field(..., description="Executive verdict on video production readiness")
    fix_tickets: List[FixTicket] = Field(default_factory=list, description="Prescriptive remediation directives")


# ============================================================================
# Verifier Client Implementation
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

        # Assemble prompt payload
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
                temperature=0.1,  # Low temperature for deterministic scoring
            ),
        )

        raw_json = response.text
        result_data = json.loads(raw_json)
        return VerificationResult(**result_data)
```

---

## 3. Verifier System Prompt Text

The following system prompt text is injected verbatim into the multimodal verification payload:

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

## 4. Weighted Score Formula & Threshold Rules

The composite video score is computed via linear combination:

$$\text{overall\_score} = (S_{\text{pi}} \times 25) + (S_{\text{bf}} \times 20) + (S_{\text{sq}} \times 20) + (S_{\text{av}} \times 15) + (S_{\text{tl}} \times 10) + (S_{\text{cq}} \times 10)$$

Where:
- $S_{\text{pi}} \in [0.0, 1.0]$: Product Integrity score
- $S_{\text{bf}} \in [0.0, 1.0]$: Brand Fidelity score
- $S_{\text{sq}} \in [0.0, 1.0]$: Scene Quality score
- $S_{\text{av}} \in [0.0, 1.0]$: AV Synchronization score
- $S_{\text{tl}} \in [0.0, 1.0]$: Text Legibility score
- $S_{\text{cq}} \in [0.0, 1.0]$: Cinematic Quality score

### Pass/Fail Invariants
1. **Score Invariant**: $\text{overall\_score} \ge 85.0$
2. **Safety Invariant**: $\forall t \in \text{fix\_tickets}, \text{severity}(t) \ne \text{"CRITICAL"}$
3. **Execution Decision**:
   $$\text{passed} = (\text{overall\_score} \ge 85.0) \land (\text{CRITICAL\_COUNT} = 0)$$

---

## 5. Autonomous Regeneration Loop with SQLite Tracking

When a video fails verification, the engine intercepts the `fix_tickets`, mutates the failed scene prompts or render configurations, and initiates targeted re-generation.

### SQLite Schema (`verification_attempts`)
```sql
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
```

### Complete Python Regeneration Loop Implementation
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        patched = json.loads(json.dumps(manifest))  # Deep copy
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
                # Global fix: append to brand negative guidelines
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
        
        recompile_pipeline_fn is a callable(patched_manifest, attempt_number) -> new_video_path
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

## 1. Complete `brand_kit.json` Schema

The `brand_kit.json` contract defines all typographic, chromatic, geometric, and tonality parameters required by the generation and compositing engines.

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
      "description": "Official legal or commercial brand display name"
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

### Corresponding Pydantic v2 Models with Custom Validators
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

## 2. Brand Asset Validation Engine

Validation guarantees that uploaded logos possess a true 8-bit alpha channel for transparent overlay via MoviePy/FFmpeg, and uploaded fonts are genuine TrueType/OpenType binaries.

```python
"""
src/ui/validators.py - Deep Validation for Brand Logos & Fonts
"""

import io
from pathlib import Path
from typing import Tuple
from PIL import Image
from fastapi import HTTPException, UploadFile


def validate_logo_image(file_bytes: bytes, filename: str) -> Tuple[int, int, float]:
    """
    Validates that the uploaded logo:
    1. Is a valid, decodable image.
    2. Has format == 'PNG'.
    3. Has mode == 'RGBA' with an active, non-opaque alpha channel.
    4. Has minimum dimensions of 128x128 px and aspect ratio between 0.1 and 10.0.
    
    Returns (width, height, aspect_ratio).
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # Verify header and integrity
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupt image file '{filename}': {str(e)}")

    # Re-open after verify() invalidated the buffer
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

    # Check for actual alpha channel transparency variations
    alpha_extrema = image.getchannel("A").getextrema()  # (min_alpha, max_alpha)
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
    Validates that the font:
    1. Ends with a valid font extension (.ttf, .otf, .woff2).
    2. Contains valid TrueType/OpenType magic byte headers.
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
    # Standard TrueType/OpenType/WOFF2 Magic Numbers:
    # 0x00010000 (TrueType)
    # 'OTTO' (OpenType with CFF)
    # 'true' (TrueType Apple)
    # 'wOF2' (WOFF2)
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
    if not template_path.exists():
        # Fallback to embedded HTML form if template file is absent
        return HTMLResponse(content=COMPLETE_ONBOARDING_HTML_FORM)
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))


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
    # 1. Resolve brand_id slug
    slug = brand_id.strip() if brand_id else re.sub(r"[^a-z0-9_-]", "", brand_name.lower().replace(" ", "_"))
    if not re.match(r"^[a-z0-9_-]{3,64}$", slug):
        raise HTTPException(
            status_code=400,
            detail="brand_id must be 3-64 characters containing only lowercase letters, digits, underscores, or hyphens."
        )

    brand_dir = VAULT_BASE_DIR / slug
    brand_dir.mkdir(parents=True, exist_ok=True)

    # 2. Validate and Persist Logo File
    logo_bytes = await logo_file.read()
    w, h, aspect_ratio = validate_logo_image(logo_bytes, logo_file.filename)
    saved_logo_path = brand_dir / "logo.png"
    saved_logo_path.write_bytes(logo_bytes)

    # 3. Validate and Persist Font File
    font_bytes = await font_file.read()
    font_ext = validate_font_file(font_bytes, font_file.filename)
    font_stem = Path(font_file.filename).stem
    saved_font_path = brand_dir / f"primary_font{font_ext}"
    saved_font_path.write_bytes(font_bytes)

    # 4. Parse CSV strings to lists
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

    # 6. Save final brand_kit.json into vault
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

## 4. Complete Onboarding HTML Form

The following responsive, dark-mode single-page HTML form is self-contained with client-side image validation, interactive color swatch previews, and asynchronous multipart submission.

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
      --accent-hover: #0284c7;
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
    .hex-val { font-family: monospace; font-size: 13px; color: var(--text-muted); }
    .dropzone {
      border: 2px dashed var(--border); border-radius: 8px; padding: 24px; text-align: center; cursor: pointer; transition: all 0.2s; background: rgba(15, 23, 42, 0.4);
    }
    .dropzone:hover, .dropzone.dragover { border-color: var(--accent); background: rgba(56, 189, 248, 0.05); }
    .dropzone p { font-size: 13px; color: var(--text-muted); margin-top: 6px; }
    .preview-box { margin-top: 12px; display: none; align-items: center; gap: 12px; background: #0f172a; padding: 10px; border-radius: 6px; }
    .preview-box img { max-height: 48px; max-width: 120px; object-fit: contain; }
    .btn-submit {
      background: linear-gradient(135deg, #0284c7, #38bdf8); color: #0f172a; font-weight: 700; border: none; border-radius: 8px;
      padding: 14px 28px; font-size: 15px; cursor: pointer; width: 100%; transition: opacity 0.2s, transform 0.1s;
    }
    .btn-submit:hover { opacity: 0.95; }
    .btn-submit:active { transform: scale(0.99); }
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
    // Color picker synchronization
    ['primary', 'secondary', 'accent'].forEach(name => {
      const picker = document.getElementById(`${name}_picker`);
      const txt = document.getElementById(`${name}_color`);
      picker.addEventListener('input', e => txt.value = e.target.value.toUpperCase());
      txt.addEventListener('input', e => {
        if (/^#[0-9A-F]{6}$/i.test(e.target.value)) picker.value = e.target.value;
      });
    });

    // Logo client preview
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

    // Font selection preview
    const fontInput = document.getElementById('font_file');
    fontInput.addEventListener('change', e => {
      const file = e.target.files[0];
      if (file) {
        document.getElementById('fontDetails').innerText = `${file.name} (${Math.round(file.size / 1024)} KB)`;
        document.getElementById('fontPreview').style.display = 'flex';
      }
    });

    // Form submission
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
        if (!res.ok) {
          throw new Error(data.detail || 'Failed to register brand');
        }
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

## 1. End-to-End Async Pipeline Function

The pipeline function executes the strict Stage 3A ➔ 3B ➔ 3C ➔ 3D ➔ Stage 4 Verifier sequence with transactional SQLite status updates and robust exception propagation.

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

# Generation sub-engines (Modules 3A-3D & 4)
from src.generation.preprocessor import run_preprocessor
from src.generation.scene_chain_engine import SceneChainEngine
from src.generation.audio_engine import generate_voiceover, detect_rhythm_beats, build_ducked_audio
from src.compositor.compositor import VideoCompositor
from src.verifier.regeneration_loop import RegenerationController

logger = logging.getLogger("orchestrator.pipeline")


# ============================================================================
# SQLite State Machine Helper
# ============================================================================

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


# ============================================================================
# Pipeline Core Callable
# ============================================================================

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

    # Load brand kit configuration
    with open(brand_kit_path, "r", encoding="utf-8") as f:
        brand_kit = json.load(f)

    try:
        # --------------------------------------------------------------------
        # STAGE 0: INITIALIZATION
        # --------------------------------------------------------------------
        update_job_status(db_path, job_id, "RUNNING", 5, "INITIALIZING")

        # --------------------------------------------------------------------
        # STAGE 3A: ASSET PREPROCESSING
        # --------------------------------------------------------------------
        update_job_status(db_path, job_id, "RUNNING", 15, "PREPROCESSING_ASSETS")
        preprocessed_png = str(scratch_dir / "product_isolated_upscaled.png")
        try:
            run_preprocessor(raw_asset_path, preprocessed_png)
        except Exception as prep_err:
            logger.warning(f"Preprocessor warning: {prep_err}. Falling back to raw asset.")
            shutil.copy(raw_asset_path, preprocessed_png)

        # --------------------------------------------------------------------
        # STAGE 3B: SCENE CHAIN GENERATION
        # --------------------------------------------------------------------
        update_job_status(db_path, job_id, "RUNNING", 35, "SCENE_CHAIN_GENERATION")
        scene_engine = SceneChainEngine(db_path=db_path)
        scene_mp4_paths = scene_engine.generate_all_scenes(
            job_id=job_id,
            manifest=manifest,
            initial_anchor_image=preprocessed_png,
            output_dir=str(job_dir / "scenes")
        )

        # --------------------------------------------------------------------
        # STAGE 3C: AUDIO SYNTHESIS & BEAT EXTRACTION
        # --------------------------------------------------------------------
        update_job_status(db_path, job_id, "RUNNING", 55, "AUDIO_SYNTHESIS")
        audio_dir = job_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Voiceover TTS
        vo_script = " ".join([s.get("voiceover_script", "") for s in manifest.get("scenes", []) if s.get("voiceover_script")])
        vo_wav_path = str(audio_dir / "voiceover.wav")
        generate_voiceover(
            script=vo_script,
            output_path=vo_wav_path,
            voice_id=brand_kit.get("voice_and_tone", {}).get("tts_voice_id", "en-US-Neural2-D"),
            speaking_rate=brand_kit.get("voice_and_tone", {}).get("reading_speed_multiplier", 1.05)
        )

        # 2. Downbeat rhythm extraction
        music_vibe = manifest.get("music_vibe", "energetic")
        tempo, beat_times = detect_rhythm_beats(vibe=music_vibe)

        # --------------------------------------------------------------------
        # STAGE 3D: COMPOSITION & ASSEMBLY
        # --------------------------------------------------------------------
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

        # --------------------------------------------------------------------
        # STAGE 4: MULTIMODAL VERIFICATION & REGENERATION LOOP
        # --------------------------------------------------------------------
        update_job_status(db_path, job_id, "RUNNING", 88, "VERIFYING")

        def recompile_subpipeline(patched_manifest: Dict[str, Any], attempt_num: int) -> str:
            """Closure passed to RegenerationController for targeted re-renders."""
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

        # --------------------------------------------------------------------
        # FINALIZATION
        # --------------------------------------------------------------------
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

The FastAPI server exposes 4 primary production endpoints backed by SQLite persistence, background execution tasks, and explicit Pydantic schemas.

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

# Register Brand UI & Onboarding Router
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
    # 1. Verify brand exists
    brand_dir = Path("storage/brand_vaults") / request.brand_id
    brand_kit_file = brand_dir / "brand_kit.json"
    if not brand_kit_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Brand '{request.brand_id}' not found. Please complete /onboarding first."
        )

    # 2. Verify raw asset exists
    if not Path(request.product_image_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"Product image file not found at '{request.product_image_path}'"
        )

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    # 3. Assemble default manifest if scenes not explicitly provided
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

    # 4. Insert row into SQLite
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

    # 5. Dispatch async background task
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
        
        # 1. Fetch Job
        cur.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        job_row = cur.fetchone()
        if not job_row:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        # 2. Fetch Verification Attempts
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

    # Patch manifest according to feedback
    scenes = manifest.get("scenes", [])
    target_idxs = request.targeted_scene_indices or list(range(len(scenes)))

    for idx in target_idxs:
        if 0 <= idx < len(scenes):
            scenes[idx]["visual_prompt"] = f"{scenes[idx]['visual_prompt']}, {request.feedback_prompt}"
            logger.info(f"[Job {job_id}] Revised scene {idx} visual prompt with feedback.")

    manifest["scenes"] = scenes

    # Update database records
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

    # Trigger async pipeline execution in background
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


# ============================================================================
# Local Development Runner
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 3. Revision Handling Logic

When an ad finishes execution but requires creative iteration (e.g. client feedback such as "make the background moodier and add more contrast to the gold trim"), `POST /jobs/{job_id}/revise` executes the following lifecycle:

1. **State Invalidation**: The endpoint guards against race conditions by asserting that `status NOT IN ('RUNNING', 'QUEUED')`.
2. **Lineage Ledger**: Records the revision request into the `job_revisions` table with a new `revision_id`, user prompt, and targeted scene array.
3. **Manifest Mutation**:
   - If `targeted_scene_indices` is specified (e.g., `[1]`), only Scene 1's visual directives and negative prompts are mutated.
   - If omitted, feedback is applied globally across all scene visual prompts.
4. **State Machine Reset**:
   - `jobs.revision_count` increments by 1.
   - `jobs.status` is reset to `'QUEUED'`.
   - `jobs.current_stage` transitions to `'QUEUED_REVISION'`.
5. **Selective Regeneration**:
   - The scene generator detects unchanged scenes and reuses existing cached MP4 clips from `storage/jobs/{job_id}/scenes/`.
   - Only modified scenes are submitted to Vertex AI Veo 2.
   - Re-compositing and verification run as normal, incrementing the attempt count.

---

## Consolidated SQLite Schema

For turnkey production setup, the SQLite schema across all engines is summarized below:

```sql
-- 1. Master Jobs Table
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

-- 2. Verification Attempts Table (Module 4)
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

-- 3. Human Revision Ledger (Module 6)
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
