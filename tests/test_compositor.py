"""
Verification tests for Module 3D: Video Compositor (src/compositor/compositor.py)
"""

import sys
import unittest
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil

# Ensure workspace src is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.compositor.compositor import VideoCompositor


class TestCompositor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.compositor = VideoCompositor()
        
        self.clip_a = Path(self.test_dir) / "clip_a.mp4"
        self.clip_b = Path(self.test_dir) / "clip_b.mp4"
        self.clip_a.write_bytes(b"VIDEO_A_BYTES")
        self.clip_b.write_bytes(b"VIDEO_B_BYTES")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_write_concat_filelist_strict_absolute_paths(self):
        filelist_out = Path(self.test_dir) / "filelist.txt"
        clips = [self.clip_a, self.clip_b]

        self.compositor.write_concat_filelist(clips, filelist_out)
        self.assertTrue(filelist_out.exists())

        lines = filelist_out.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        
        # Verify strict format: file 'absolute/path/to/clip.mp4'
        abs_a = self.clip_a.resolve().as_posix()
        abs_b = self.clip_b.resolve().as_posix()
        self.assertEqual(lines[0], f"file '{abs_a}'")
        self.assertEqual(lines[1], f"file '{abs_b}'")

    @patch("subprocess.run")
    def test_direct_concat_command(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(returncode=0)
        filelist_txt = Path(self.test_dir) / "filelist.txt"
        joined_raw = Path(self.test_dir) / "joined_raw.mp4"

        self.compositor.concatenate_clips_direct([self.clip_a, self.clip_b], joined_raw, filelist_txt)

        expected_cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(filelist_txt),
            "-c", "copy",
            str(joined_raw),
            "-y"
        ]
        mock_subprocess.assert_called_once_with(
            expected_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    @patch("src.compositor.compositor.VideoCompositor.get_media_duration", return_value=5.0)
    @patch("subprocess.run")
    def test_cross_dissolve_command(self, mock_subprocess, mock_dur):
        mock_subprocess.return_value = MagicMock(returncode=0)
        out_mp4 = Path(self.test_dir) / "cross_dissolve_out.mp4"

        # Duration 0.3s -> offset = 5.0 - 0.3 = 4.700
        self.compositor.render_cross_dissolve(self.clip_a, self.clip_b, out_mp4, duration=0.3)

        expected_filter = "[0:v][1:v]xfade=transition=fade:duration=0.3:offset=4.700[v];[0:a][1:a]acrossfade=d=0.3[a]"
        expected_cmd = [
            "ffmpeg",
            "-i", str(self.clip_a),
            "-i", str(self.clip_b),
            "-filter_complex", expected_filter,
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-crf", "18",
            str(out_mp4),
            "-y"
        ]
        mock_subprocess.assert_called_once_with(
            expected_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    @patch("src.compositor.compositor.VideoCompositor.get_media_duration", return_value=5.0)
    @patch("subprocess.run")
    def test_whip_pan_right_uses_wipeleft(self, mock_subprocess, mock_dur):
        mock_subprocess.return_value = MagicMock(returncode=0)
        out_mp4 = Path(self.test_dir) / "whip_pan_out.mp4"

        self.compositor.render_whip_pan_right(self.clip_a, self.clip_b, out_mp4, duration=0.3)

        expected_filter = "[0:v][1:v]xfade=transition=wipeleft:duration=0.3:offset=4.700[v];[0:a][1:a]acrossfade=d=0.3[a]"
        expected_cmd = [
            "ffmpeg",
            "-i", str(self.clip_a),
            "-i", str(self.clip_b),
            "-filter_complex", expected_filter,
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-crf", "18",
            str(out_mp4),
            "-y"
        ]
        mock_subprocess.assert_called_once_with(
            expected_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    def test_moviepy_logo_overlay_exact_1_0_3_api(self):
        mock_moviepy = MagicMock()
        mock_editor = MagicMock()
        mock_moviepy.editor = mock_editor

        mock_video = MagicMock()
        mock_video.duration = 15.0
        mock_video.h = 1920
        mock_video.fps = 30
        mock_editor.VideoFileClip.return_value = mock_video

        mock_logo_clip = MagicMock()
        mock_logo_clip.set_duration.return_value = mock_logo_clip
        mock_logo_clip.set_opacity.return_value = mock_logo_clip
        mock_logo_clip.resize.return_value = mock_logo_clip
        mock_logo_clip.set_position.return_value = mock_logo_clip
        mock_editor.ImageClip.return_value = mock_logo_clip

        mock_composite = MagicMock()
        mock_editor.CompositeVideoClip.return_value = mock_composite

        with patch.dict("sys.modules", {"moviepy": mock_moviepy, "moviepy.editor": mock_editor}):
            logo_img = Path(self.test_dir) / "brand_logo.png"
            logo_img.write_bytes(b"LOGO_BYTES")
            out_branded = Path(self.test_dir) / "branded.mp4"

            self.compositor.overlay_brand_logo_moviepy(self.clip_a, logo_img, out_branded)

            mock_editor.VideoFileClip.assert_called_once_with(str(self.clip_a))
            mock_editor.ImageClip.assert_called_once_with(str(logo_img))
            mock_logo_clip.set_duration.assert_called_once_with(15.0)
            mock_logo_clip.set_opacity.assert_called_once_with(0.85)
            mock_logo_clip.resize.assert_called_once_with(height=int(1920 * 0.08))
            mock_logo_clip.set_position.assert_called_once_with(('right', 'top'), relative=True)
            mock_editor.CompositeVideoClip.assert_called_once_with([mock_video, mock_logo_clip])

    @patch("subprocess.run")
    def test_audio_ducking_command(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(returncode=0)
        vo_wav = Path(self.test_dir) / "voiceover.wav"
        out_mp4 = Path(self.test_dir) / "output_with_audio.mp4"

        self.compositor.duck_audio(self.clip_a, vo_wav, out_mp4, music_volume_ducked=0.15)

        expected_cmd = [
            "ffmpeg",
            "-i", str(self.clip_a),
            "-i", str(vo_wav),
            "-filter_complex", "[0:a]volume=0.15[music];[music][1:a]amix=inputs=2:duration=first:dropout_transition=2[a]",
            "-map", "0:v",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(out_mp4),
            "-y"
        ]
        mock_subprocess.assert_called_once_with(
            expected_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    @patch("subprocess.run")
    def test_final_master_export_command(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(returncode=0)
        composed_mp4 = Path(self.test_dir) / "composed.mp4"
        final_mp4 = Path(self.test_dir) / "final.mp4"

        self.compositor.export_final_master(composed_mp4, final_mp4)

        expected_cmd = [
            "ffmpeg",
            "-i", str(composed_mp4),
            "-vcodec", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-acodec", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-s", "1080x1920",
            "-r", "30",
            str(final_mp4),
            "-y"
        ]
        mock_subprocess.assert_called_once_with(
            expected_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )


if __name__ == "__main__":
    unittest.main()
