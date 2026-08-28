"""
Second Brain — Media Processing & Transcription Module
======================================================
Handles:
  1. Audio extraction from .mp4 using moviepy
  2. Transcription via Groq Whisper API (whisper-large-v3-turbo)
  3. Auto-generated 3-word title via LiteLLM
  4. Aggressive temp file cleanup
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import litellm
from groq import Groq

litellm.suppress_debug_info = True

# Accepted file types
AUDIO_TYPES = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".wma"}
VIDEO_TYPES = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
ALL_MEDIA = AUDIO_TYPES | VIDEO_TYPES


def get_groq_client() -> Groq:
    """Create a Groq client from the GROQ_API_KEY env var."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment or .env file")
    return Groq(api_key=api_key)


def extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio from a video file using moviepy.
    Returns the path to the temporary .mp3 file.
    Caller is responsible for cleanup.
    """
    from moviepy.editor import VideoFileClip

    tmp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_audio.close()

    try:
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(
            tmp_audio.name,
            fps=16000,     # Whisper optimal sample rate
            nbytes=2,      # 16-bit
            codec="libmp3lame",
            bitrate="64k", # Low bitrate for speed
            logger=None,   # Suppress moviepy output
        )
        clip.close()
    except Exception as e:
        # Cleanup on failure
        safe_remove(tmp_audio.name)
        raise RuntimeError(f"Audio extraction failed: {e}")

    return tmp_audio.name


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe an audio file using Groq's Whisper API.
    Returns {"text": str, "duration": float, "language": str}.
    """
    client = get_groq_client()
    file_ext = Path(audio_path).suffix.lower()

    # Map file extensions to MIME types
    mime_map = {
        ".mp3": "audio/mpeg",
        ".mp4": "audio/mp4",
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }

    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=(Path(audio_path).name, f, mime_map.get(file_ext, "audio/mpeg")),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )

    return {
        "text": result.text,
        "duration": getattr(result, "duration", 0),
        "language": getattr(result, "language", "unknown"),
    }


async def generate_title(transcript: str, max_chars: int = 500) -> str:
    """
    Use LiteLLM to generate a concise 3-word title from a transcript snippet.
    Falls back to a simple extraction if the LLM call fails.
    """
    snippet = transcript[:max_chars]
    try:
        import asyncio
        response = await litellm.acompletion(
            model="gemini/gemini-2.0-flash",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate exactly 3 words that summarize this transcript. "
                        "No quotes, no punctuation, no explanation. Just 3 words."
                    ),
                },
                {"role": "user", "content": snippet},
            ],
            temperature=0.3,
            max_tokens=20,
        )
        title = response.choices[0].message.content.strip().strip('"').strip("'")
        words = title.split()
        if len(words) > 3:
            title = " ".join(words[:3])
        return title
    except Exception:
        # Fallback: extract first 3 meaningful words from transcript
        words = [w for w in transcript.split() if len(w) > 2 and w.isalpha()]
        return " ".join(words[:3]) if words else "Untitled Media"


def format_duration(seconds: float) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    if seconds <= 0:
        return "unknown"
    seconds = int(seconds)
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}:{s:02d}"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{h}:{m:02d}:{s:02d}"


def safe_remove(path: str) -> None:
    """Safely delete a temp file, ignoring errors."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


async def process_media(file_path: str, file_name: str) -> dict:
    """
    Full pipeline: extract audio → transcribe → generate title.
    Returns {"title": str, "transcript": str, "duration": str, "file_type": str, "language": str}.
    Aggressively cleans up temp files.
    """
    ext = Path(file_name).suffix.lower()
    tmp_audio = None

    try:
        # Step 1: Extract audio if video
        if ext in VIDEO_TYPES:
            tmp_audio = extract_audio_from_video(file_path)
            audio_path = tmp_audio
        else:
            audio_path = file_path

        # Step 2: Transcribe
        result = transcribe_audio(audio_path)

        # Step 3: Generate title
        title = await generate_title(result["text"])

        # Step 4: Format duration
        duration = format_duration(result.get("duration", 0))

        return {
            "title": title,
            "transcript": result["text"],
            "duration": duration,
            "file_type": ext.lstrip("."),
            "language": result.get("language", "unknown"),
        }

    finally:
        # Aggressive cleanup
        if tmp_audio:
            safe_remove(tmp_audio)
