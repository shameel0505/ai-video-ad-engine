"""
Module 3B: Scene Chain Continuity Engine
Path: src/generation/scene_chain_engine.py

Manages multi-scene generative continuity loops using Google Vertex AI Veo 2.
Extracts tail frames at t=4.8s using FFmpeg for scene-to-scene visual chaining,
tracks state in SQLite, and handles timeouts (180s) and corrupt clip fallbacks.
"""

import os
import shutil
import sqlite3
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import concurrent.futures

logger = logging.getLogger("generation.scene_chain_engine")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class SceneChainEngine:
    """
    Orchestrates continuous multi-scene video generation via Google Vertex AI Veo 2.
    Maintains visual continuity by feeding Scene N's tail frame (extracted at 4.8s)
    into Scene N+1's image prompt input.
    """

    DEFAULT_DB_PATH = Path("storage/generation_jobs.db")

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_sqlite()

    def _init_sqlite(self):
        """Initializes SQLite tables for job tracking and per-scene progression."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generation_jobs (
                    job_id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_scenes INTEGER NOT NULL,
                    current_scene INTEGER DEFAULT 0,
                    error_log TEXT,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scene_outputs (
                    job_id TEXT NOT NULL,
                    scene_idx INTEGER NOT NULL,
                    clip_path TEXT,
                    tail_frame_path TEXT,
                    prompt TEXT,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP,
                    PRIMARY KEY (job_id, scene_idx)
                )
            """)
            conn.commit()

    def insert_job(self, job_id: str, project_id: str, total_scenes: int):
        """Records a new multi-scene generation job in SQLite."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO generation_jobs 
                (job_id, project_id, status, total_scenes, current_scene, error_log, created_at, updated_at)
                VALUES (?, ?, 'INITIALIZING', ?, 0, '', ?, ?)
                """,
                (job_id, project_id, total_scenes, now, now)
            )
            conn.commit()
        logger.info("Job %s initialized in SQLite database.", job_id)

    def update_scene_progress(
        self,
        job_id: str,
        scene_idx: int,
        clip_path: Optional[str],
        tail_frame_path: Optional[str],
        prompt: str,
        status: str,
        error_msg: str = ""
    ):
        """Updates per-scene generation output and parent job progress."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO scene_outputs
                (job_id, scene_idx, clip_path, tail_frame_path, prompt, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, scene_idx, clip_path, tail_frame_path, prompt, status, now)
            )
            cursor.execute(
                """
                UPDATE generation_jobs
                SET current_scene = ?,
                    status = ?,
                    error_log = error_log || ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (scene_idx, f"SCENE_{scene_idx}_{status}", f" [Scene {scene_idx}: {error_msg}]" if error_msg else "", now, job_id)
            )
            conn.commit()
        logger.info("Job %s progress updated: scene %d -> %s", job_id, scene_idx, status)

    def mark_generation_complete(self, job_id: str):
        """Flags job status as GENERATION_COMPLETE upon all scenes finishing."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE generation_jobs
                SET status = 'GENERATION_COMPLETE',
                    updated_at = ?
                WHERE job_id = ?
                """,
                (now, job_id)
            )
            conn.commit()
        logger.info("Job %s marked GENERATION_COMPLETE.", job_id)

    def extract_tail_frame(self, clip_path: str, output_png: str) -> str:
        """
        Extracts tail frame at second 4.8 of a 5.0s clip using FFmpeg.
        Command:
            ffmpeg -ss 4.8 -i {clip_path} -vframes 1 -q:v 2 {output_png} -y
        """
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-ss", "4.8",
            "-i", str(clip_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_png),
            "-y"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg tail frame extraction failed: {result.stderr}")
        return output_png

    def is_clip_valid(self, clip_path: str) -> bool:
        """Verifies if the generated MP4 clip is non-corrupt using OpenCV."""
        try:
            import cv2
            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened():
                return False
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return frame_count > 0
        except Exception as exc:
            logger.warning("OpenCV validation error on %s: %s", clip_path, exc)
            return False

    def generate_single_scene_veo2(
        self,
        combined_prompt: str,
        start_frame_path: str,
        output_path: str,
        gcs_output_path: str
    ) -> str:
        """
        Generates 5-second 9:16 clip using Google Vertex AI Veo 2 model.
        Supports MOCK_MODE for local zero-cost simulation.
        """
        # MOCK STUB check
        if os.getenv("MOCK_MODE") == "true":
            logger.info("[MOCK_MODE] Copying sample clip to %s", output_path)
            mock_src = Path("storage/mocks/sample_clip_5s.mp4")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if not mock_src.exists():
                # Create dummy mock clip if not pre-populated
                mock_src.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    "ffmpeg", "-f", "lavfi", "-i", "color=c=black:s=1080x1920:d=5",
                    "-c:v", "libx264", "-t", "5", "-pix_fmt", "yuv420p", str(mock_src), "-y"
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            shutil.copy(str(mock_src), output_path)
            return output_path

        # Live Vertex AI Veo 2 API call
        import vertexai
        from vertexai.preview.vision_models import VideoGenerationModel, Image as VisionImage
        from google.cloud import storage

        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            raise ValueError("GCP_PROJECT_ID environment variable is missing.")

        vertexai.init(project=project_id, location="us-central1")
        model = VideoGenerationModel.from_pretrained("veo-2.0-generate-001")
        
        videos = model.generate_video(
            prompt=combined_prompt,
            image=VisionImage.load_from_file(start_frame_path),
            duration_seconds=5,
            aspect_ratio="9:16",
            output_gcs_uri=gcs_output_path
        )

        def download_from_gcs(gcs_uri: str, local_dest: str) -> str:
            # Format: gs://bucket_name/path/to/blob
            uri_parts = gcs_uri.replace("gs://", "").split("/", 1)
            bucket_name, blob_name = uri_parts[0], uri_parts[1]
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            Path(local_dest).parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(local_dest)
            return local_dest

        local_path = download_from_gcs(videos[0].uri, output_path)
        return local_path

    def run_scene_chain(
        self,
        job_id: str,
        project_id: str,
        scenes: List[Dict[str, Any]],
        initial_anchor_image_path: str,
        output_dir: str
    ) -> List[str]:
        """
        Executes complete multi-scene continuity state machine.
        
        State loop:
        1. Scene 0 begins with initial product anchor image.
        2. Scene N generates 5s clip via Veo 2 with 180s timeout guard.
        3. Validates clip integrity via cv2.VideoCapture.
        4. Extracts tail frame at 4.8s via FFmpeg.
        5. Hands off tail frame as start_frame_path for Scene N+1.
        6. Updates SQLite records after each scene.
        7. Marks GENERATION_COMPLETE upon termination.
        """
        total_scenes = len(scenes)
        self.insert_job(job_id, project_id, total_scenes)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        current_anchor = initial_anchor_image_path
        generated_clips: List[str] = []

        gcs_bucket = os.getenv("GCS_OUTPUT_BUCKET", "gs://ai-video-ad-engine-renders")

        for idx, scene in enumerate(scenes):
            clip_filename = f"{job_id}_scene_{idx}.mp4"
            clip_path = str(out_path / clip_filename)
            tail_frame_path = str(out_path / f"{job_id}_tail_{idx}.png")
            gcs_dest = f"{gcs_bucket}/{job_id}/{clip_filename}"

            combined_prompt = f"{scene.get('visual_prompt', '')}, camera motion: {scene.get('camera_motion', 'cinematic dolly')}"
            logger.info("Executing Scene %d/%d (Anchor: %s)", idx, total_scenes, current_anchor)

            # Veo 2 invocation with 180s strict timeout
            clip_generated = False
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.generate_single_scene_veo2,
                    combined_prompt,
                    current_anchor,
                    clip_path,
                    gcs_dest
                )
                try:
                    future.result(timeout=180.0)
                    clip_generated = True
                except concurrent.futures.TimeoutError:
                    logger.warning("Veo 2 timed out after 180s on Scene %d. Skipping scene; holding anchor.", idx)
                    self.update_scene_progress(job_id, idx, None, current_anchor, combined_prompt, "TIMEOUT_SKIPPED", "Veo 2 exceeded 180s timeout")
                except Exception as exc:
                    logger.warning("Veo 2 execution failed on Scene %d: %s", idx, exc)
                    self.update_scene_progress(job_id, idx, None, current_anchor, combined_prompt, "FAILED", str(exc))

            if clip_generated:
                # Corrupt clip detection
                if not self.is_clip_valid(clip_path):
                    logger.warning("Scene %d clip corrupt (cv2 check failed). Using fallback anchor.", idx)
                    self.update_scene_progress(job_id, idx, None, current_anchor, combined_prompt, "CORRUPT_FALLBACK", "cv2.VideoCapture check failed")
                else:
                    # Successfully validated clip
                    try:
                        self.extract_tail_frame(clip_path, tail_frame_path)
                        current_anchor = tail_frame_path
                        generated_clips.append(clip_path)
                        self.update_scene_progress(job_id, idx, clip_path, tail_frame_path, combined_prompt, "COMPLETED")
                    except Exception as exc:
                        logger.warning("Failed tail frame extraction on Scene %d: %s. Preserving previous anchor.", idx, exc)
                        current_anchor = current_anchor  # keep previous
                        generated_clips.append(clip_path)
                        self.update_scene_progress(job_id, idx, clip_path, current_anchor, combined_prompt, "TAIL_EXTRACT_ERROR", str(exc))

        self.mark_generation_complete(job_id)
        return generated_clips
