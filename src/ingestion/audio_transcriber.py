"""
YouTube Audio Transcriber using yt-dlp and Google GenAI.

Downloads audio from YouTube videos and transcribes using Gemini.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import yt_dlp
from google import genai
from google.genai import types


MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
    ".opus": "audio/opus",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
}

TRANSCRIPT_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "segments": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "start_time": types.Schema(type=types.Type.STRING),
                    "end_time": types.Schema(type=types.Type.STRING),
                    "speaker": types.Schema(type=types.Type.STRING),
                    "language": types.Schema(type=types.Type.STRING),
                    "content": types.Schema(type=types.Type.STRING),
                },
                required=["start_time", "end_time", "content"],
            ),
        ),
    },
    required=["segments"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TranscriberConfig:
    """Configuration for the audio transcriber."""
    audio_dir: Path = Path("data/audio")
    model: str = "gemini-2.5-flash"
    retries: int = 3


@dataclass
class TranscriptionResult:
    """Result of a transcription including usage statistics."""
    transcription: dict
    input_tokens: int
    output_tokens: int
    total_tokens: int


class AudioTranscriber:
    """Transcribe YouTube videos using yt-dlp and Google GenAI."""

    def __init__(self, config: TranscriberConfig | None = None):
        self.config = config or TranscriberConfig()
        self.config.audio_dir.mkdir(parents=True, exist_ok=True)
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        """Lazy-load the GenAI client."""
        if self._client is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def count_audio_tokens(self, audio_path: Path) -> int:
        """Count tokens for audio file. ~32 tokens per second of audio."""
        client = self._get_client()
        logger.info(f"Counting tokens for: {audio_path}")
        audio_file = client.files.upload(file=str(audio_path))
        response = client.models.count_tokens(
            model=self.config.model,
            contents=[audio_file]
        )
        logger.info(f"Audio tokens: {response.total_tokens} (~{response.total_tokens // 32}s)")
        return response.total_tokens

    def download_audio(self, url: str) -> tuple[Path, str]:
        """
        Download audio from a YouTube video.

        Args:
            url: YouTube video URL

        Returns:
            Tuple of (audio file path, video title)
        """
        logger.info(f"Downloading audio from: {url}")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(self.config.audio_dir / "%(title).100s.%(ext)s"),
            "retries": self.config.retries,
            "fragment_retries": self.config.retries,
            "socket_timeout": 30,
            "restrictfilenames": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            video_title = info.get("title", "unknown")

        audio_path = Path(filename)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Audio downloaded: {audio_path}")
        return audio_path, video_title

    def transcribe_audio(self, audio_path: Path) -> TranscriptionResult:
        """
        Transcribe an audio file using Google GenAI with timestamps.

        Args:
            audio_path: Path to the audio file

        Returns:
            TranscriptionResult with transcription dict and token usage stats
        """
        logger.info(f"Transcribing audio: {audio_path}")

        client = self._get_client()

        # Upload the audio file
        logger.info("Uploading audio to Google GenAI...")
        audio_file = client.files.upload(file=str(audio_path))
        logger.info(f"Upload complete: {audio_file.name}")

        # Wait for file to be processed
        while audio_file.state.name == "PROCESSING":
            logger.info("Waiting for file to be processed...")
            time.sleep(2)
            audio_file = client.files.get(name=audio_file.name)

        if audio_file.state.name != "ACTIVE":
            raise RuntimeError(f"File processing failed: {audio_file.state.name}")

        # Generate transcription with timestamps
        logger.info("Generating transcription with timestamps...")
        response = client.models.generate_content(
            model=self.config.model,
            contents=[
                types.Part.from_uri(file_uri=audio_file.uri, mime_type=audio_file.mime_type),
                """Generate a detailed verbatim transcript of this audio.

Requirements:
- Transcribe all spoken words exactly as said (no paraphrasing)
- Preserve the original language (do not translate)
- Identify different speakers (use "Speaker 1", "Speaker 2", etc.)
- Detect the language of each segment
- Provide timestamps in HH:MM:SS,mmm format (SRT compatible)
- Create segments of 5-10 seconds each

For each segment provide: start_time, end_time, speaker, language, content."""
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TRANSCRIPT_SCHEMA,
            ),
        )

        transcription = json.loads(response.text)
        segment_count = len(transcription.get("segments", []))

        # Extract usage metadata for cost tracking
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0
        total_tokens = usage.total_token_count if usage else 0

        logger.info(
            f"Transcription complete: {segment_count} segments, "
            f"{input_tokens:,} input tokens, {output_tokens:,} output tokens"
        )

        return TranscriptionResult(
            transcription=transcription,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def _normalize_timestamp(self, timestamp: str) -> str:
        """
        Normalize timestamp to standard SRT format: HH:MM:SS,mmm

        Handles various Gemini output formats:
        - MM:SS:mmm -> 00:MM:SS,mmm
        - MM:SS,mmm -> 00:MM:SS,mmm
        - HH:MM:SS:mmm -> HH:MM:SS,mmm
        - HH:MM:SS,mmm -> HH:MM:SS,mmm (already valid)
        """
        # Replace all separators with colons for parsing
        parts = timestamp.replace(",", ":").split(":")

        if len(parts) == 3:
            # Format: MM:SS:mmm -> add hours
            mm, ss, mmm = parts
            return f"00:{mm.zfill(2)}:{ss.zfill(2)},{mmm.zfill(3)}"
        elif len(parts) == 4:
            # Format: HH:MM:SS:mmm
            hh, mm, ss, mmm = parts
            return f"{hh.zfill(2)}:{mm.zfill(2)}:{ss.zfill(2)},{mmm.zfill(3)}"
        else:
            # Fallback: return as-is with comma separator
            return timestamp.replace(":", ",", timestamp.count(":") - 1) if ":" in timestamp else timestamp

    def _convert_to_srt(self, transcription: dict) -> str:
        """
        Convert JSON transcription to SRT format.

        Args:
            transcription: Dict with segments

        Returns:
            SRT formatted string
        """
        srt_lines = []
        for i, segment in enumerate(transcription.get("segments", []), start=1):
            start_time = self._normalize_timestamp(segment.get("start_time", "00:00:00,000"))
            end_time = self._normalize_timestamp(segment.get("end_time", "00:00:00,000"))
            speaker = segment.get("speaker", "")
            content = segment.get("content", "")

            # Add speaker prefix if available
            text = f"[{speaker}] {content}" if speaker else content

            srt_lines.append(str(i))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text)
            srt_lines.append("")  # Empty line between entries

        return "\n".join(srt_lines)

    def save_transcription(self, transcription: dict, base_path: Path) -> tuple[Path, Path]:
        """
        Save transcription as both JSON and SRT files.

        Args:
            transcription: Dict with segments
            base_path: Base path for output files (extension will be replaced)

        Returns:
            Tuple of (json_path, srt_path)
        """
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = base_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(transcription, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        logger.info(f"JSON saved: {json_path}")

        # Save SRT
        srt_path = base_path.with_suffix(".srt")
        srt_path.write_text(self._convert_to_srt(transcription), encoding="utf-8")
        logger.info(f"SRT saved: {srt_path}")

        return json_path, srt_path

    def process(
        self,
        url: str,
        output_path: Path | None = None,
        delete_audio: bool = False,
    ) -> tuple[Path, Path, TranscriptionResult]:
        """
        Full pipeline: download audio, transcribe, and save.

        Args:
            url: YouTube video URL
            output_path: Optional base path for transcription (extension ignored)
            delete_audio: Whether to delete the audio file after transcription

        Returns:
            Tuple of (json_path, srt_path, transcription_result)
        """
        # Download audio
        audio_path, _ = self.download_audio(url)

        try:
            # Transcribe
            result = self.transcribe_audio(audio_path)

            # Determine base output path (same directory as audio)
            if output_path is None:
                output_path = audio_path.with_suffix("")

            # Save transcription (both JSON and SRT)
            json_path, srt_path = self.save_transcription(result.transcription, output_path)
            return json_path, srt_path, result

        finally:
            # Clean up audio if requested
            if delete_audio and audio_path.exists():
                audio_path.unlink()
                logger.info(f"Deleted audio file: {audio_path}")


def main():
    """CLI entry point for the audio transcriber."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and transcribe YouTube videos using Gemini"
    )
    parser.add_argument(
        "--url",
        required=True,
        help="YouTube video URL to transcribe",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Base output path for transcription (default: data/transcripts/{title})",
    )
    parser.add_argument(
        "--delete-audio",
        action="store_true",
        help="Delete the audio file after transcription",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/audio"),
        help="Directory to save audio and transcript files",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use for transcription",
    )

    args = parser.parse_args()

    config = TranscriberConfig(
        audio_dir=args.audio_dir,
        model=args.model,
    )

    transcriber = AudioTranscriber(config=config)

    try:
        json_path, srt_path, result = transcriber.process(
            url=args.url,
            output_path=args.output,
            delete_audio=args.delete_audio,
        )
        print(f"\nTranscription saved to:\n  JSON: {json_path}\n  SRT:  {srt_path}")
        print(f"\nToken usage:\n  Input:  {result.input_tokens:,}\n  Output: {result.output_tokens:,}")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
