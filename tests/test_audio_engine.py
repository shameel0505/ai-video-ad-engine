"""
Verification tests for Module 3C: Audio Generation & Processing Engine (src/generation/audio_engine.py)
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

from src.generation.audio_engine import AudioEngine


class TestAudioEngine(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.music_dir = Path(self.test_dir) / "music_library"
        self.audio_engine = AudioEngine(music_library_root=self.music_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_invalid_vibe_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.audio_engine.select_background_music("dubstep_techno")
        self.assertIn("Invalid vibe 'dubstep_techno'", str(ctx.exception))

    def test_empty_vibe_folder_raises_value_error(self):
        # Energetic folder exists but is empty
        energetic_dir = self.music_dir / "energetic"
        energetic_dir.mkdir(parents=True, exist_ok=True)

        with self.assertRaises(ValueError) as ctx:
            self.audio_engine.select_background_music("energetic")
        self.assertIn(
            "No music tracks found for vibe 'energetic' in storage/music_library/energetic/. Please add .mp3 files.",
            str(ctx.exception)
        )

    def test_valid_vibe_selection(self):
        luxury_dir = self.music_dir / "luxury"
        luxury_dir.mkdir(parents=True, exist_ok=True)
        track = luxury_dir / "track_01.mp3"
        track.write_bytes(b"MOCK_MP3_DATA")

        selected = self.audio_engine.select_background_music("luxury")
        self.assertEqual(selected.resolve(), track.resolve())

    def test_librosa_beat_extraction(self):
        mock_librosa = MagicMock()
        mock_librosa.load.return_value = (MagicMock(), 44100)
        mock_librosa.beat.beat_track.return_value = (124.5, [100, 200, 300])
        mock_librosa.frames_to_time.return_value = MagicMock(tolist=lambda: [0.5, 1.0, 1.5])

        with patch.dict("sys.modules", {"librosa": mock_librosa}):
            tempo, beats = self.audio_engine.extract_beats("test_track.mp3")
            self.assertEqual(tempo, 124.5)
            self.assertEqual(beats, [0.5, 1.0, 1.5])
            mock_librosa.load.assert_called_once_with("test_track.mp3", sr=None)

    def test_google_cloud_tts_exact_call(self):
        mock_google = MagicMock()
        mock_cloud = MagicMock()
        mock_tts = MagicMock()
        mock_google.cloud = mock_cloud
        mock_cloud.texttospeech = mock_tts

        mock_client = MagicMock()
        mock_tts.TextToSpeechClient.return_value = mock_client
        mock_tts.SsmlVoiceGender.MALE = "MALE"
        mock_tts.AudioEncoding.LINEAR16 = "LINEAR16"

        mock_response = MagicMock()
        mock_response.audio_content = b"WAV_AUDIO_RAW_BINARY_CONTENT"
        mock_client.synthesize_speech.return_value = mock_response

        with patch.dict("sys.modules", {
            "google": mock_google,
            "google.cloud": mock_cloud,
            "google.cloud.texttospeech": mock_tts
        }):
            out_wav = Path(self.test_dir) / "output_voice.wav"
            res_path = self.audio_engine.generate_voiceover("Stop scrolling. Look at this.", out_wav)

            self.assertTrue(res_path.exists())
            self.assertEqual(res_path.read_bytes(), b"WAV_AUDIO_RAW_BINARY_CONTENT")

            # Verify parameters
            mock_tts.SynthesisInput.assert_called_once_with(text="Stop scrolling. Look at this.")
            mock_tts.VoiceSelectionParams.assert_called_once_with(
                language_code='en-US',
                name='en-US-Neural2-D',
                ssml_gender='MALE'
            )
            mock_tts.AudioConfig.assert_called_once_with(
                audio_encoding='LINEAR16',
                speaking_rate=1.05
            )

    @patch("subprocess.run")
    def test_ffmpeg_spectral_noise_cancel(self, mock_subprocess):
        mock_subprocess.return_value = MagicMock(returncode=0)
        in_mp4 = Path(self.test_dir) / "input.mp4"
        out_wav = Path(self.test_dir) / "cleaned.wav"

        self.audio_engine.clean_audio_noise(in_mp4, out_wav)

        expected_cmd = [
            "ffmpeg",
            "-i", str(in_mp4),
            "-af", "highpass=f=200,lowpass=f=3000,afftdn=nf=-25",
            str(out_wav),
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
