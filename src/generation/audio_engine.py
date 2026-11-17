"""
Module 3C: Audio Generation & Processing Engine
Path: src/generation/audio_engine.py

Handles sound design, vibe-based BGM selection, Librosa rhythmic beat detection,
Google Cloud Text-to-Speech voiceover generation, and FFmpeg spectral de-noising.
"""

import os
import random
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple, Union, Optional

logger = logging.getLogger("generation.audio_engine")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class AudioEngine:
    """
    Audio generation, beat synchronization, and voiceover engine.
    """

    ALLOWED_VIBES = {"energetic", "luxury", "chill", "dramatic", "upbeat", "asmr"}

    def __init__(self, music_library_root: Optional[Union[str, Path]] = None):
        self.music_library_root = (
            Path(music_library_root) if music_library_root else Path("storage/music_library")
        )

    def select_background_music(self, vibe: str) -> Path:
        """
        Selects a random .mp3 track from storage/music_library/{vibe}/.
        
        Validation:
            vibe must be one of energetic, luxury, chill, dramatic, upbeat, asmr.
            If folder is empty, raises ValueError with a descriptive message.
        """
        vibe_normalized = vibe.lower().strip()
        if vibe_normalized not in self.ALLOWED_VIBES:
            raise ValueError(
                f"Invalid vibe '{vibe}'. Must be one of: {sorted(list(self.ALLOWED_VIBES))}"
            )

        vibe_folder = self.music_library_root / vibe_normalized
        if not vibe_folder.exists():
            vibe_folder.mkdir(parents=True, exist_ok=True)

        mp3_tracks = list(vibe_folder.glob("*.mp3"))
        if not mp3_tracks:
            raise ValueError(
                f"No music tracks found for vibe '{vibe_normalized}' in "
                f"storage/music_library/{vibe_normalized}/. Please add .mp3 files."
            )

        selected_track = random.choice(mp3_tracks)
        logger.info("Selected background music track: %s (vibe: %s)", selected_track.name, vibe_normalized)
        return selected_track

    def extract_beats(self, music_path: Union[str, Path]) -> Tuple[float, List[float]]:
        """
        Extracts tempo (BPM) and beat timestamps using Librosa.
        
        Exact call:
            import librosa
            y, sr = librosa.load(music_path, sr=None)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        """
        import librosa

        music_path_str = str(music_path)
        logger.info("Extracting rhythmic beats via Librosa from %s", music_path_str)
        y, sr = librosa.load(music_path_str, sr=None)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        # Handle scalar or array tempo
        tempo_val = float(tempo[0]) if hasattr(tempo, "__getitem__") else float(tempo)
        logger.info("Beat analysis complete: Tempo=%.2f BPM, Total Beats=%d", tempo_val, len(beat_times))
        return tempo_val, beat_times

    def generate_voiceover(
        self,
        script: str,
        output_wav_path: Union[str, Path],
        speaking_rate: float = 1.05
    ) -> Path:
        """
        Generates studio-grade voiceover audio using Google Cloud Text-to-Speech (Neural2).
        
        Exact call:
            from google.cloud import texttospeech
            client = texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=script)
            voice = texttospeech.VoiceSelectionParams(
                language_code='en-US',
                name='en-US-Neural2-D',
                ssml_gender=texttospeech.SsmlVoiceGender.MALE
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                speaking_rate=1.05
            )
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            write response.audio_content to WAV file
        """
        output_wav_path = Path(output_wav_path)
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        if os.getenv("MOCK_MODE") == "true":
            logger.info("[MOCK_MODE] Synthesizing mock voiceover tone to %s", output_wav_path)
            # Create a 3s silent/sine WAV as mock
            cmd = [
                "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                "-ac", "1", "-ar", "24000", str(output_wav_path), "-y"
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_wav_path

        from google.cloud import texttospeech

        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=script)
        voice = texttospeech.VoiceSelectionParams(
            language_code='en-US',
            name='en-US-Neural2-D',
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=speaking_rate
        )

        logger.info("Calling Google Cloud TTS with Neural2-D voice (speed: %.2f)", speaking_rate)
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        with open(output_wav_path, "wb") as out:
            out.write(response.audio_content)

        logger.info("Voiceover written to WAV: %s (%d bytes)", output_wav_path, len(response.audio_content))
        return output_wav_path

    def clean_audio_noise(
        self,
        input_media_path: Union[str, Path],
        output_wav_path: Union[str, Path]
    ) -> Path:
        """
        Removes background noise and sub-bass rumble using FFmpeg audio filters:
            ffmpeg -i input.mp4 -af 'highpass=f=200,lowpass=f=3000,afftdn=nf=-25' cleaned.wav -y
        
        Filter chain breakdown:
        - highpass=f=200: Attenuates sub-bass rumble, HVAC hum, wind noise, and proximity thumps.
        - lowpass=f=3000: Attenuates ultrasonic hiss and high-frequency interference outside voice band.
        - afftdn=nf=-25: Adaptive Fast Fourier Transform de-noiser with a noise floor of -25dB.
        Avoids external heavy ONNX dependencies required by arnndn.
        """
        input_media_path = Path(input_media_path)
        output_wav_path = Path(output_wav_path)
        output_wav_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg",
            "-i", str(input_media_path),
            "-af", "highpass=f=200,lowpass=f=3000,afftdn=nf=-25",
            str(output_wav_path),
            "-y"
        ]

        logger.info("Executing FFmpeg spectral audio clean filter...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio cleaning failed: {result.stderr}")

        logger.info("Cleaned audio saved to: %s", output_wav_path)
        return output_wav_path
