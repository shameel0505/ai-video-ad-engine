# AI Video Ad Engine: Modules 3A, 3B, 3C, 3D Technical Specification

---

## Architecture Overview: Stage 3 Production Engine

Stage 3 transforms multimodal creative directives and raw product assets into broadcast-ready 9:16 vertical video ads. It consists of four tightly coupled sub-engines:

```
[Raw Asset] 
     │
     ▼
┌──────────────────────────────────────────────┐
│  3A: Preprocessor                            │
│  - rembg (u2net subject extraction)          │
│  - Replicate Real-ESRGAN (4x super-res)      │
└──────────────────────┬───────────────────────┘
                       │ Staged Canvas (1080x1920)
                       ▼
┌──────────────────────────────────────────────┐
│  3B: Scene Chain Engine                      │
│  - Veo 2 Vertex AI Sequential Generator      │
│  - Tail Frame Extraction (t=4.8s)            │
│  - SQLite State Machine & Corrupt Fallback   │
└──────────────────────┬───────────────────────┘
                       │ Raw Scene MP4s
                       ▼
┌──────────────────────────────────────────────┐
│  3C: Audio Engine                            │
│  - Vibe-based Music Library                  │
│  - Librosa Rhythmic Beat Detection           │
│  - Google Cloud TTS (Neural2-D Voiceover)    │
│  - FFmpeg Spectral Noise Cancellation        │
└──────────────────────┬───────────────────────┘
                       │ Dynamic Audio Layers (Music + VO + SFX)
                       ▼
┌──────────────────────────────────────────────┐
│  3D: Compositor                              │
│  - Strict Concat Demuxer Protocol            │
│  - FFmpeg Transitions (Cross-dissolve/Whip)  │
│  - MoviePy 1.0.3 Brand Logo Overlay          │
│  - Audio Ducking & J-Cut SFX Bridges         │
│  - Master MP4 Export (1080x1920 @ 30fps)     │
└──────────────────────────────────────────────┘
```

---

## Module 3A: Preprocessor (`src/generation/preprocessor.py`)

### 1. Purpose & Pipeline
The Preprocessor isolates the physical product from messy background environments and upscales low-resolution client photos into high-fidelity starting visual anchors for video diffusion models.

### 2. Implementation Specifications

#### Background Removal (`rembg`)
- **Model**: `u2net` (General salient object detection, ~170MB ONNX model).
- **Storage Location**: `~/.u2net/u2net.onnx`.
- **First-Run Warmup Warning**:
  ```bash
  # Must execute once post-installation to prevent runtime timeout during job execution:
  python -c 'from rembg import remove'
  ```
- **Exact Call**:
  ```python
  from rembg import remove

  with open(input_path, 'rb') as inp:
      output_bytes = remove(inp.read())
  with open(output_png, 'wb') as out:
      out.write(output_bytes)
  ```
- **Error Handling**:
  If `rembg` crashes or fails during execution (e.g. corrupted input image, ONNX runtime failure, CUDA out-of-memory), catch the exception, log `WARNING`, and write the original raw image bytes directly to `output_png`.

#### Super-Resolution Upscaling (Replicate Real-ESRGAN)
- **Model Endpoint**: `nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b`
- **Parameters**: `scale: 4`, `face_enhance: False`.
- **Exact Call**:
  ```python
  import replicate
  import httpx

  output = replicate.run(
      'nightmareai/real-esrgan:42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b',
      input={'image': open(path, 'rb'), 'scale': 4, 'face_enhance': False}
  )
  target_url = output if isinstance(output, str) else output[0]
  response = httpx.get(target_url, timeout=60.0)
  response.raise_for_status()
  with open(local_path, 'wb') as f:
      f.write(response.content)
  ```
- **Error Handling & Retry Policy**:
  - Wrap in a retry block with `max_attempts = 2`.
  - On exception (rate limits, 5xx errors, timeouts), wait 2.0 seconds (`time.sleep(2)`) and retry once.
  - If the retry fails, log a `WARNING` and fall back to the cleaned, un-upscaled image.

---

## Module 3B: Scene Chain Engine (`src/generation/scene_chain_engine.py`)

### 1. Purpose & Continuity Protocol
Video diffusion models generate individual 5-second shots without inherent scene-to-scene awareness. The Scene Chain Engine maintains continuity across scenes:
1. Scene 0 starts with the preprocessed product image.
2. Scene $i$ ($i > 0$) receives the tail frame ($t = 4.8\text{s}$) of Scene $i - 1$ as its image prompt input.

### 2. State Machine & SQLite Schema

```
[PENDING] ──> [INITIALIZING] ──> [SCENE_0_GENERATING] ──> [SCENE_0_COMPLETED]
                                                                  │
                                            ┌─────────────────────┘
                                            ▼
                                  [SCENE_1_GENERATING] ──> [SCENE_1_COMPLETED]
                                                                  │
                                            ┌─────────────────────┘
                                            ▼
                                 [GENERATION_COMPLETE]
```

#### Database Schema
```sql
CREATE TABLE IF NOT EXISTS generation_jobs (
    job_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    status TEXT NOT NULL,
    total_scenes INTEGER NOT NULL,
    current_scene INTEGER DEFAULT 0,
    error_log TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scene_outputs (
    job_id TEXT NOT NULL,
    scene_idx INTEGER NOT NULL,
    clip_path TEXT,
    tail_frame_path TEXT,
    prompt TEXT,
    status TEXT NOT NULL,
    created_at TIMESTAMP,
    PRIMARY KEY (job_id, scene_idx)
);
```

### 3. Google Vertex AI Veo 2 Integration
- **Model Identifier**: `veo-2.0-generate-001`
- **Location**: `us-central1`
- **Exact Call**:
  ```python
  import os
  import vertexai
  from vertexai.preview.vision_models import VideoGenerationModel, Image as VisionImage

  vertexai.init(project=os.getenv('GCP_PROJECT_ID'), location='us-central1')
  model = VideoGenerationModel.from_pretrained('veo-2.0-generate-001')
  videos = model.generate_video(
      prompt=combined_prompt,
      image=VisionImage.load_from_file(start_frame_path),
      duration_seconds=5,
      aspect_ratio='9:16',
      output_gcs_uri=gcs_output_path
  )
  local_path = download_from_gcs(videos[0].uri)
  ```
- **Local Mock Stub**:
  ```python
  if os.getenv('MOCK_MODE') == 'true':
      shutil.copy('storage/mocks/sample_clip_5s.mp4', output_path)
      return output_path
  ```

### 4. Tail Frame FFmpeg Extraction
Extracts frame at $t = 4.8\text{s}$ to allow a 200–300ms transition buffer:
```bash
ffmpeg -ss 4.8 -i {clip_path} -vframes 1 -q:v 2 {output_png} -y
```

### 5. Error Handling & Guardrails
- **Veo 2 Timeout (180s)**:
  Generation is executed within a thread executor with a strict 180.0-second timeout. If exceeded:
  - Log `WARNING`.
  - Skip current scene generation.
  - Retain the previous scene's tail frame as the visual anchor.
- **Corrupt Clip Verification**:
  Validate downloaded clips using `cv2.VideoCapture`:
  ```python
  import cv2
  cap = cv2.VideoCapture(clip_path)
  if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 0:
      logger.warning("Clip corrupt; using previous tail frame fallback.")
  ```

---

## Module 3C: Audio Engine (`src/generation/audio_engine.py`)

### 1. Purpose
Selects brand-appropriate background music, detects downbeats for rhythmic scene cuts, synthesizes voiceover tracks, and strips background audio noise.

### 2. Music Library Structure
- **Storage Layout**: `storage/music_library/{vibe}/*.mp3`
- **Supported Vibes**: `energetic`, `luxury`, `chill`, `dramatic`, `upbeat`, `asmr`.
- **Validation Rule**:
  If the specified folder does not contain any `.mp3` files:
  ```python
  raise ValueError(f"No music tracks found for vibe '{vibe}' in storage/music_library/{vibe}/. Please add .mp3 files.")
  ```

### 3. Librosa Beat Extraction
- **Exact Call**:
  ```python
  import librosa

  y, sr = librosa.load(music_path, sr=None)
  tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
  beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
  ```
  Returns `tempo` (BPM) and float timestamp array `beat_times` for transition synchronization.

### 4. Google Cloud Text-to-Speech (Neural2)
- **Voice Model**: `en-US-Neural2-D` (Male, authoritative commercial tone).
- **Encoding**: Linear 16-bit PCM (`LINEAR16`).
- **Speaking Rate**: `1.05` (+5% acceleration for social media attention retention).
- **Exact Call**:
  ```python
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
  with open(output_wav_path, "wb") as out:
      out.write(response.audio_content)
  ```

### 5. FFmpeg Spectral Noise Cancellation
Replaces `arnndn` (which requires external neural model weights) with FFmpeg's native DSP filter chain:
```bash
ffmpeg -i input.mp4 -af 'highpass=f=200,lowpass=f=3000,afftdn=nf=-25' cleaned.wav -y
```
- `highpass=f=200`: Cuts plosives, microphone rumble, and HVAC sub-bass hum.
- `lowpass=f=3000`: Removes high-frequency hiss outside fundamental voice frequencies.
- `afftdn=nf=-25`: Adaptive FFT de-noiser tracking static background noise floor down to -25dB.

---

## Module 3D: Compositor (`src/compositor/`)

### 1. Concat Demuxer Protocol
- **Filelist Format**:
  Every entry in `filelist.txt` must be absolute paths wrapped in single quotes:
  ```
  file '/absolute/path/to/clip_0.mp4'
  file '/absolute/path/to/clip_1.mp4'
  file '/absolute/path/to/clip_2.mp4'
  ```
- **Direct Concat Command**:
  ```bash
  ffmpeg -f concat -safe 0 -i filelist.txt -c copy joined_raw.mp4 -y
  ```

### 2. Transitions

#### Cross-Dissolve
Transitions video and audio simultaneously:
```bash
ffmpeg -i clip_a.mp4 -i clip_b.mp4 \
  -filter_complex '[0:v][1:v]xfade=transition=fade:duration=0.3:offset={offset}[v];[0:a][1:a]acrossfade=d=0.3[a]' \
  -map '[v]' -map '[a]' -c:v libx264 -crf 18 out.mp4 -y
```
*Note*: `offset = clip_a_duration - 0.3` (e.g. $5.0 - 0.3 = 4.7\text{s}$).

#### Whip-Pan Right
In FFmpeg `xfade`, panning the camera to the right slides the incoming scene from the right to the left. Hence, use `wipeleft`:
```bash
ffmpeg -i clip_a.mp4 -i clip_b.mp4 \
  -filter_complex '[0:v][1:v]xfade=transition=wipeleft:duration=0.3:offset={offset}[v];[0:a][1:a]acrossfade=d=0.3[a]' \
  -map '[v]' -map '[a]' -c:v libx264 -crf 18 out.mp4 -y
```

### 3. Brand Logo Overlay (MoviePy 1.0.3 API)
Complies strictly with `moviepy==1.0.3` API (does not use `moviepy>=2.0` syntax):
```python
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

video = VideoFileClip(video_path)
logo = (
    ImageClip(logo_path)
    .set_duration(video.duration)
    .set_opacity(0.85)
    .resize(height=int(video.h * 0.08))
)
logo = logo.set_position(('right', 'top'), relative=True)
final = CompositeVideoClip([video, logo])
final.write_videofile(output_path, codec='libx264', audio_codec='aac')
```

### 4. Audio Ducking
Lowers background music to 15% volume during voiceover:
```bash
ffmpeg -i video.mp4 -i voiceover.wav \
  -filter_complex '[0:a]volume=0.15[music];[music][1:a]amix=inputs=2:duration=first:dropout_transition=2[a]' \
  -map 0:v -map '[a]' -c:v copy -c:a aac -b:a 192k output_with_audio.mp4 -y
```

### 5. J-Cut SFX Bridge
In a J-Cut, the audio from the incoming scene (e.g., transition whoosh) starts before the visual cut occurs:
- If visual cut is at $t = 5.0\text{s}$, a $0.25\text{s}$ pre-roll means the SFX must begin at $t = 4.75\text{s}$.
- Using `-itsoffset`:
  ```bash
  # Pre-rolls SFX by offsetting stream start time:
  ffmpeg -i video.mp4 -itsoffset 4.75 -i sfx_whoosh.wav \
    -filter_complex '[0:a][1:a]amix=inputs=2:duration=first[a]' \
    -map 0:v -map '[a]' -c:v copy -c:a aac -b:a 192k output_jcut.mp4 -y
  ```
- Using `adelay` filter:
  ```bash
  ffmpeg -i video.mp4 -i sfx_whoosh.wav \
    -filter_complex '[1:a]adelay=4750|4750[sfx];[0:a][sfx]amix=inputs=2:duration=first[a]' \
    -map 0:v -map '[a]' -c:v copy -c:a aac -b:a 192k output_jcut.mp4 -y
  ```

### 6. Master Video Export
Encodes final 9:16 vertical video at 1080x1920 30fps with Web/FastStart optimization:
```bash
ffmpeg -i composed_video.mp4 -vcodec libx264 -preset slow -crf 18 \
  -acodec aac -b:a 192k -movflags +faststart -s 1080x1920 -r 30 final.mp4 -y
```

---

## Verification Test Matrix

All 4 modules have dedicated unit and integration tests under `tests/`:

| Module | Test File | Test Cases Covered |
| :--- | :--- | :--- |
| **3A Preprocessor** | `tests/test_preprocessor.py` | 1. Normal `rembg` background removal.<br>2. `rembg` crash fallback to raw image bytes.<br>3. Normal Replicate Real-ESRGAN 4x upscaling.<br>4. Replicate failure retry logic & fallback to unscaled image. |
| **3B Scene Chain** | `tests/test_scene_chain_engine.py` | 1. SQLite state machine lifecycle (`INITIALIZING` -> `COMPLETED`).<br>2. FFmpeg tail frame extraction command verification.<br>3. `MOCK_MODE` 5s video simulation.<br>4. OpenCV corrupt clip detection and fallback anchor retention. |
| **3C Audio Engine** | `tests/test_audio_engine.py` | 1. Invalid vibe validation.<br>2. Empty vibe folder `ValueError` guard.<br>3. Valid track selection.<br>4. Librosa tempo and beat time extraction.<br>5. Google Cloud TTS exact parameter assertion.<br>6. FFmpeg spectral noise cancellation filter execution. |
| **3D Compositor** | `tests/test_compositor.py` | 1. Strict `filelist.txt` absolute path formatting.<br>2. Direct FFmpeg concat stream-copy command.<br>3. Cross-dissolve `xfade` with dynamic duration offset calculation.<br>4. Whip-pan right using `wipeleft`.<br>5. MoviePy 1.0.3 logo overlay API compatibility.<br>6. FFmpeg audio ducking command.<br>7. Final master export parameters. |
