"""
Module 3D: Video Compositor
Path: src/compositor/compositor.py

Assembles video clips using FFmpeg concat protocol, applies xfade transitions
(cross_dissolve and whip_pan_right/wipeleft), overlays brand logos via MoviePy 1.0.3,
performs audio ducking, builds J-cut SFX bridges, and exports production-ready 1080x1920 30fps MP4.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List, Union, Optional

logger = logging.getLogger("compositor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class VideoCompositor:
    """
    Programmatic video assembly engine coordinating FFmpeg and MoviePy 1.0.3.
    """

    @staticmethod
    def get_media_duration(file_path: Union[str, Path]) -> float:
        """Queries precise media duration in seconds via ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed to read duration for {file_path}: {result.stderr}")
        return float(result.stdout.strip())

    @staticmethod
    def write_concat_filelist(clip_paths: List[Union[str, Path]], filelist_path: Union[str, Path]) -> Path:
        """
        Generates FFmpeg concat demuxer filelist.
        Format constraint:
            Each line must be formatted as: file 'absolute/path/to/clip.mp4'
            Must strictly use absolute paths and single quotes.
        """
        filelist_path = Path(filelist_path)
        filelist_path.parent.mkdir(parents=True, exist_ok=True)

        with open(filelist_path, "w", encoding="utf-8") as f:
            for clip in clip_paths:
                abs_path = Path(clip).resolve().as_posix()
                f.write(f"file '{abs_path}'\n")

        logger.info("Generated filelist.txt at: %s with %d clips", filelist_path, len(clip_paths))
        return filelist_path

    def concatenate_clips_direct(
        self,
        clip_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        filelist_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Direct stream-copy concatenation for clips with identical encoding/codecs/resolution.
        Command:
            ffmpeg -f concat -safe 0 -i filelist.txt -c copy joined_raw.mp4 -y
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not filelist_path:
            filelist_path = output_path.parent / f"{output_path.stem}_filelist.txt"

        self.write_concat_filelist(clip_paths, filelist_path)

        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(filelist_path),
            "-c", "copy",
            str(output_path),
            "-y"
        ]

        logger.info("Executing direct FFmpeg concat copy...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        return output_path

    def render_cross_dissolve(
        self,
        clip_a_path: Union[str, Path],
        clip_b_path: Union[str, Path],
        output_path: Union[str, Path],
        duration: float = 0.3
    ) -> Path:
        """
        Renders a cross-dissolve transition between two clips using xfade and acrossfade.
        
        Offset calculation:
            offset = clip_a_duration - duration (e.g. 5.0 - 0.3 = 4.7s)

        Command:
            ffmpeg -i clip_a.mp4 -i clip_b.mp4 \
              -filter_complex '[0:v][1:v]xfade=transition=fade:duration=0.3:offset={offset}[v];[0:a][1:a]acrossfade=d=0.3[a]' \
              -map '[v]' -map '[a]' -c:v libx264 -crf 18 out.mp4 -y
        """
        clip_a_dur = self.get_media_duration(clip_a_path)
        offset = max(0.0, clip_a_dur - duration)

        cmd = [
            "ffmpeg",
            "-i", str(clip_a_path),
            "-i", str(clip_b_path),
            "-filter_complex",
            f"[0:v][1:v]xfade=transition=fade:duration={duration}:offset={offset:.3f}[v];[0:a][1:a]acrossfade=d={duration}[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-crf", "18",
            str(output_path),
            "-y"
        ]

        logger.info("Executing FFmpeg cross_dissolve (offset=%.3f, duration=%.2f)...", offset, duration)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg cross_dissolve failed: {result.stderr}")

        return Path(output_path)

    def render_whip_pan_right(
        self,
        clip_a_path: Union[str, Path],
        clip_b_path: Union[str, Path],
        output_path: Union[str, Path],
        duration: float = 0.3
    ) -> Path:
        """
        Renders a whip-pan right transition between two clips.
        
        Note:
            In FFmpeg xfade transition naming, moving camera pan to the right
            reveals incoming content moving to the left: use xfade=transition=wipeleft.

        Command:
            ffmpeg -i clip_a.mp4 -i clip_b.mp4 \
              -filter_complex '[0:v][1:v]xfade=transition=wipeleft:duration=0.3:offset={offset}[v];[0:a][1:a]acrossfade=d=0.3[a]' \
              -map '[v]' -map '[a]' -c:v libx264 -crf 18 out.mp4 -y
        """
        clip_a_dur = self.get_media_duration(clip_a_path)
        offset = max(0.0, clip_a_dur - duration)

        cmd = [
            "ffmpeg",
            "-i", str(clip_a_path),
            "-i", str(clip_b_path),
            "-filter_complex",
            f"[0:v][1:v]xfade=transition=wipeleft:duration={duration}:offset={offset:.3f}[v];[0:a][1:a]acrossfade=d={duration}[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-crf", "18",
            str(output_path),
            "-y"
        ]

        logger.info("Executing FFmpeg whip_pan_right via wipeleft (offset=%.3f)...", offset)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg whip_pan_right failed: {result.stderr}")

        return Path(output_path)

    def overlay_brand_logo_moviepy(
        self,
        video_path: Union[str, Path],
        logo_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Overlays client brand logo onto video using MoviePy 1.0.3 API strictly.
        
        Exact API call:
            from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
            video = VideoFileClip(video_path)
            logo = ImageClip(logo_path).set_duration(video.duration).set_opacity(0.85).resize(height=int(video.h * 0.08))
            logo = logo.set_position(('right', 'top'), relative=True)
            final = CompositeVideoClip([video, logo])
        """
        from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

        video_path_str = str(video_path)
        logo_path_str = str(logo_path)
        output_path_str = str(output_path)

        logger.info("Applying logo overlay with MoviePy 1.0.3...")
        video = VideoFileClip(video_path_str)
        logo = (
            ImageClip(logo_path_str)
            .set_duration(video.duration)
            .set_opacity(0.85)
            .resize(height=int(video.h * 0.08))
        )
        logo = logo.set_position(('right', 'top'), relative=True)
        final = CompositeVideoClip([video, logo])

        final.write_videofile(
            output_path_str,
            codec="libx264",
            audio_codec="aac",
            fps=video.fps or 30,
            logger=None
        )

        video.close()
        final.close()
        return Path(output_path)

    def duck_audio(
        self,
        video_path: Union[str, Path],
        voiceover_path: Union[str, Path],
        output_path: Union[str, Path],
        music_volume_ducked: float = 0.15
    ) -> Path:
        """
        Ducks background music to 15% volume during voiceover speech.
        
        Command:
            ffmpeg -i video.mp4 -i voiceover.wav \
              -filter_complex '[0:a]volume=0.15[music];[music][1:a]amix=inputs=2:duration=first:dropout_transition=2[a]' \
              -map 0:v -map '[a]' -c:v copy -c:a aac -b:a 192k output_with_audio.mp4 -y
        """
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", str(voiceover_path),
            "-filter_complex",
            f"[0:a]volume={music_volume_ducked}[music];[music][1:a]amix=inputs=2:duration=first:dropout_transition=2[a]",
            "-map", "0:v",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
            "-y"
        ]

        logger.info("Executing FFmpeg audio ducking (music volume: %.2f)...", music_volume_ducked)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg audio ducking failed: {result.stderr}")

        return Path(output_path)

    def render_j_cut_sfx_bridge(
        self,
        video_path: Union[str, Path],
        sfx_path: Union[str, Path],
        output_path: Union[str, Path],
        cut_point_sec: float,
        pre_roll_sec: float = 0.25
    ) -> Path:
        """
        Implements a J-Cut sound design bridge for scene transitions.
        
        Concept:
            In professional film editing, a J-Cut occurs when the incoming scene's sound
            (e.g., whoosh, swoosh, impact, or dialogue) leads in and begins before
            the visual cut takes place.
            
        Offset approach using -itsoffset:
            -itsoffset delays the start of the following input stream relative to timestamp 0.
            To place the SFX whoosh centered across a scene cut at timestamp T, the SFX
            must start playing at (T - pre_roll_sec) = (T - 0.25s).
            Using -itsoffset:
                ffmpeg -i video.mp4 -itsoffset {lead_in_timestamp} -i sfx_whoosh.wav ...
            Alternatively via adelay filter:
                adelay={start_ms}|{start_ms} delays the SFX audio buffer by start_ms.
        """
        start_time_sec = max(0.0, cut_point_sec - pre_roll_sec)
        delay_ms = int(start_time_sec * 1000)

        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", str(sfx_path),
            "-filter_complex",
            f"[1:a]adelay={delay_ms}|{delay_ms},volume=0.8[sfx];[0:a][sfx]amix=inputs=2:duration=first[a]",
            "-map", "0:v",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
            "-y"
        ]

        logger.info("Rendering J-Cut SFX bridge: Cut=%.2fs, Pre-roll=%.2fs (SFX start=%.2fs)", cut_point_sec, pre_roll_sec, start_time_sec)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg J-Cut SFX bridge failed: {result.stderr}")

        return Path(output_path)

    def export_final_master(
        self,
        composed_video_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> Path:
        """
        Exports the master 9:16 high-conversion video ad ready for social distribution.
        
        Exact export command:
            ffmpeg -i composed_video.mp4 -vcodec libx264 -preset slow -crf 18 \
              -acodec aac -b:a 192k -movflags +faststart -s 1080x1920 -r 30 final.mp4 -y
        """
        cmd = [
            "ffmpeg",
            "-i", str(composed_video_path),
            "-vcodec", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-acodec", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-s", "1080x1920",
            "-r", "30",
            str(output_path),
            "-y"
        ]

        logger.info("Executing final master FFmpeg export (1080x1920, 30fps, CRF 18, +faststart)...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg master export failed: {result.stderr}")

        return Path(output_path)
