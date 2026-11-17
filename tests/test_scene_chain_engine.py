"""
Verification tests for Module 3B: Scene Chain Continuity Engine (src/generation/scene_chain_engine.py)
"""

import sys
import os
import unittest
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import sqlite3

# Ensure workspace src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation.scene_chain_engine import SceneChainEngine


class TestSceneChainEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test_generation.db"
        self.engine = SceneChainEngine(db_path=self.db_path)
        
        # Create a mock anchor image
        self.anchor_img = Path(self.test_dir) / "anchor_0.png"
        self.anchor_img.write_bytes(b"MOCK_PNG_IMAGE_BYTES")

        # Ensure mock clip exists
        self.mock_clip = Path("storage/mocks/sample_clip_5s.mp4")
        self.mock_clip.parent.mkdir(parents=True, exist_ok=True)
        self.mock_clip.write_bytes(b"MOCK_SAMPLE_CLIP_DATA")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_sqlite_lifecycle_operations(self):
        job_id = "job_test_123"
        project_id = "proj_xyz"
        total_scenes = 3

        # 1. Insert job
        self.engine.insert_job(job_id, project_id, total_scenes)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT job_id, status, total_scenes FROM generation_jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], job_id)
            self.assertEqual(row[1], "INITIALIZING")
            self.assertEqual(row[2], 3)

        # 2. Update progress
        self.engine.update_scene_progress(
            job_id=job_id,
            scene_idx=0,
            clip_path="storage/scene_0.mp4",
            tail_frame_path="storage/tail_0.png",
            prompt="Close up shot of perfume bottle",
            status="COMPLETED"
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT scene_idx, status FROM scene_outputs WHERE job_id = ?", (job_id,))
            scene_row = cursor.fetchone()
            self.assertEqual(scene_row[0], 0)
            self.assertEqual(scene_row[1], "COMPLETED")

        # 3. Mark complete
        self.engine.mark_generation_complete(job_id)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM generation_jobs WHERE job_id = ?", (job_id,))
            status_row = cursor.fetchone()
            self.assertEqual(status_row[0], "GENERATION_COMPLETE")

    @patch("subprocess.run")
    def test_tail_frame_ffmpeg_command(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(returncode=0)
        clip_path = "storage/test_clip.mp4"
        tail_png = str(Path(self.test_dir) / "tail.png")

        self.engine.extract_tail_frame(clip_path, tail_png)
        expected_cmd = [
            "ffmpeg",
            "-ss", "4.8",
            "-i", str(clip_path),
            "-vframes", "1",
            "-q:v", "2",
            str(tail_png),
            "-y"
        ]
        mock_subprocess.assert_called_once_with(
            expected_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def test_mock_mode_execution(self):
        # Set MOCK_MODE environment variable
        with patch.dict(os.environ, {"MOCK_MODE": "true"}):
            scenes = [
                {"visual_prompt": "Scene 1: Reveal", "camera_motion": "pan right"},
                {"visual_prompt": "Scene 2: Zoom in", "camera_motion": "zoom in"}
            ]

            with patch.object(self.engine, "is_clip_valid", return_value=True), \
                 patch.object(self.engine, "extract_tail_frame", side_effect=lambda c, o: Path(o).write_bytes(b"TAIL_PNG")):

                out_clips = self.engine.run_scene_chain(
                    job_id="job_mock_456",
                    project_id="proj_mock",
                    scenes=scenes,
                    initial_anchor_image_path=str(self.anchor_img),
                    output_dir=self.test_dir
                )

                self.assertEqual(len(out_clips), 2)
                # Verify DB status
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT status FROM generation_jobs WHERE job_id = ?", ("job_mock_456",))
                    self.assertEqual(cursor.fetchone()[0], "GENERATION_COMPLETE")

    def test_corrupt_clip_fallback(self):
        scenes = [{"visual_prompt": "Scene 1: Luxury watch"}]

        with patch.dict(os.environ, {"MOCK_MODE": "true"}), \
             patch.object(self.engine, "is_clip_valid", return_value=False):

            out_clips = self.engine.run_scene_chain(
                job_id="job_corrupt_test",
                project_id="proj_corrupt",
                scenes=scenes,
                initial_anchor_image_path=str(self.anchor_img),
                output_dir=self.test_dir
            )

            # Corrupt clip should be rejected, resulting in 0 valid clips added
            self.assertEqual(len(out_clips), 0)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT status FROM scene_outputs WHERE job_id = ?", ("job_corrupt_test",))
                self.assertEqual(cursor.fetchone()[0], "CORRUPT_FALLBACK")


if __name__ == "__main__":
    unittest.main()
