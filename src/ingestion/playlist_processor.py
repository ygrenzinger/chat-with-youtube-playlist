"""
YouTube Playlist Processor.

Downloads and transcribes all videos from a YouTube playlist,
tracking progress via CSV to enable resume and skip already processed videos.
"""

import csv
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yt_dlp

from src.ingestion.audio_transcriber import AudioTranscriber, TranscriberConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Information about a video in the playlist."""
    video_id: str
    url: str
    title: str
    status: str = "pending"
    error: str = ""
    processed_at: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


# Gemini API pricing per 1M tokens (audio input rate)
# Source: https://ai.google.dev/gemini-api/docs/pricing
GEMINI_PRICING = {
    "gemini-2.5-flash": {"audio_input": 1.00, "output": 2.50},
    "gemini-2.5-flash-preview-05-20": {"audio_input": 1.00, "output": 2.50},
    "gemini-2.0-flash": {"audio_input": 0.70, "output": 0.40},
    "gemini-2.0-flash-exp": {"audio_input": 0.70, "output": 0.40},
    "gemini-2.5-pro": {"audio_input": 2.50, "output": 15.00},
    "gemini-2.5-pro-preview-05-06": {"audio_input": 2.50, "output": 15.00},
}


def get_pricing(model: str) -> tuple[float, float]:
    """Get pricing for a model. Returns (audio_input_per_million, output_per_million)."""
    # Try exact match first, then prefix match
    if model in GEMINI_PRICING:
        p = GEMINI_PRICING[model]
        return p["audio_input"], p["output"]

    # Fallback: match by prefix (e.g., "gemini-2.5-flash-xxx" -> "gemini-2.5-flash")
    for key in GEMINI_PRICING:
        if model.startswith(key):
            p = GEMINI_PRICING[key]
            return p["audio_input"], p["output"]

    # Default to 2.5-flash pricing if unknown
    logger.warning(f"Unknown model '{model}', using gemini-2.5-flash pricing")
    return 1.00, 2.50


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for given token counts."""
    audio_rate, output_rate = get_pricing(model)
    input_cost = input_tokens * audio_rate / 1_000_000
    output_cost = output_tokens * output_rate / 1_000_000
    return round(input_cost + output_cost, 6)


@dataclass
class PlaylistConfig:
    """Configuration for the playlist processor."""
    csv_path: Path = Path("data/playlist_progress.csv")
    audio_dir: Path = Path("data/audio")
    model: str = "gemini-2.5-flash"
    delay_between_videos: float = 2.0
    delete_audio: bool = False
    max_videos: int | None = 20  # None = unlimited (use --no-limit)


class PlaylistProcessor:
    """Process all videos from a YouTube playlist with progress tracking."""

    CSV_FIELDNAMES = [
        "video_id", "url", "title", "status", "error", "processed_at",
        "input_tokens", "output_tokens", "cost_usd"
    ]

    def __init__(self, config: PlaylistConfig | None = None):
        self.config = config or PlaylistConfig()
        self.config.audio_dir.mkdir(parents=True, exist_ok=True)
        self.config.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._shutdown_requested = False
        self._running_cost = 0.0
        self._running_input_tokens = 0
        self._running_output_tokens = 0
        self._videos_processed = 0
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup graceful shutdown on SIGINT/SIGTERM."""
        def signal_handler(signum, frame):
            logger.info("\nShutdown requested. Finishing current video...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def extract_playlist_videos(self, playlist_url: str) -> list[VideoInfo]:
        """
        Extract video metadata from a YouTube playlist.

        Args:
            playlist_url: URL of the YouTube playlist

        Returns:
            List of VideoInfo objects
        """
        logger.info(f"Fetching playlist metadata from: {playlist_url}")

        ydl_opts = {
            "extract_flat": True,
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)

        if not info:
            raise ValueError(f"Could not extract playlist info from: {playlist_url}")

        playlist_title = info.get("title", "Unknown Playlist")
        entries = info.get("entries", [])

        if not entries:
            raise ValueError(f"No videos found in playlist: {playlist_url}")

        videos = []
        for entry in entries:
            if entry is None:
                continue

            video_id = entry.get("id", "")
            if not video_id:
                continue

            videos.append(VideoInfo(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                title=entry.get("title", "Unknown Title"),
            ))

        logger.info(f"Found {len(videos)} videos in playlist: {playlist_title}")
        return videos

    def _read_csv(self) -> dict[str, VideoInfo]:
        """Read existing CSV and return dict keyed by video_id."""
        if not self.config.csv_path.exists():
            return {}

        videos = {}
        with open(self.config.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                videos[row["video_id"]] = VideoInfo(
                    video_id=row["video_id"],
                    url=row["url"],
                    title=row["title"],
                    status=row["status"],
                    error=row.get("error", ""),
                    processed_at=row.get("processed_at", ""),
                    input_tokens=int(row.get("input_tokens", 0) or 0),
                    output_tokens=int(row.get("output_tokens", 0) or 0),
                    cost_usd=float(row.get("cost_usd", 0) or 0),
                )
        return videos

    def _write_csv(self, videos: dict[str, VideoInfo]):
        """Write videos dict to CSV."""
        with open(self.config.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDNAMES)
            writer.writeheader()
            for video in videos.values():
                writer.writerow({
                    "video_id": video.video_id,
                    "url": video.url,
                    "title": video.title,
                    "status": video.status,
                    "error": video.error,
                    "processed_at": video.processed_at,
                    "input_tokens": video.input_tokens,
                    "output_tokens": video.output_tokens,
                    "cost_usd": video.cost_usd,
                })

    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title to match yt-dlp's restrictfilenames output."""
        # Match yt-dlp's restrictfilenames behavior
        result = []
        for char in title:
            if char.isalnum() or char in "-_.":
                result.append(char)
            elif char in " ":
                result.append("_")
        return "".join(result)[:100]

    def _check_transcript_exists(self, title: str) -> bool:
        """Check if transcript files already exist for a video."""
        sanitized = self._sanitize_filename(title)
        json_path = self.config.audio_dir / f"{sanitized}.json"
        srt_path = self.config.audio_dir / f"{sanitized}.srt"
        return json_path.exists() and srt_path.exists()

    def init_csv(self, playlist_videos: list[VideoInfo]) -> dict[str, VideoInfo]:
        """
        Initialize or update CSV with playlist videos.

        Preserves existing status for known videos.
        Adds new videos as pending.
        Marks videos with existing transcripts as done.

        Returns:
            Updated videos dict
        """
        existing = self._read_csv()

        for video in playlist_videos:
            if video.video_id in existing:
                # Check if transcript files exist but status is not done
                if existing[video.video_id].status != "done":
                    if self._check_transcript_exists(video.title):
                        existing[video.video_id].status = "done"
                        existing[video.video_id].processed_at = datetime.now().isoformat()
                        logger.info(f"Found existing transcript: {video.title}")
            else:
                # New video - check if transcript already exists
                if self._check_transcript_exists(video.title):
                    video.status = "done"
                    video.processed_at = datetime.now().isoformat()
                    logger.info(f"Found existing transcript: {video.title}")
                existing[video.video_id] = video

        self._write_csv(existing)
        return existing

    def get_pending_videos(self, videos: dict[str, VideoInfo]) -> list[VideoInfo]:
        """Get list of videos that need processing."""
        return [v for v in videos.values() if v.status not in ("done",)]

    def _update_video_status(
        self,
        videos: dict[str, VideoInfo],
        video_id: str,
        status: str,
        error: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ):
        """Update a video's status and save CSV."""
        if video_id in videos:
            videos[video_id].status = status
            videos[video_id].error = error
            if status == "done":
                videos[video_id].processed_at = datetime.now().isoformat()
                videos[video_id].input_tokens = input_tokens
                videos[video_id].output_tokens = output_tokens
                videos[video_id].cost_usd = calculate_cost(
                    self.config.model, input_tokens, output_tokens
                )
            self._write_csv(videos)

    def process_playlist(self, playlist_url: str) -> dict[str, VideoInfo]:
        """
        Process all videos in a playlist.

        Args:
            playlist_url: URL of the YouTube playlist

        Returns:
            Final videos dict with updated statuses
        """
        # Extract playlist and initialize CSV
        playlist_videos = self.extract_playlist_videos(playlist_url)
        videos = self.init_csv(playlist_videos)

        return self._process_videos(videos)

    def process_from_csv(self) -> dict[str, VideoInfo]:
        """
        Resume processing from existing CSV without fetching playlist.

        Returns:
            Final videos dict with updated statuses
        """
        videos = self._read_csv()
        if not videos:
            raise ValueError(f"No videos found in CSV: {self.config.csv_path}")

        return self._process_videos(videos)

    def _process_videos(self, videos: dict[str, VideoInfo]) -> dict[str, VideoInfo]:
        """Process pending videos from the videos dict."""
        pending = self.get_pending_videos(videos)
        total = len(videos)
        done_count = sum(1 for v in videos.values() if v.status == "done")

        if not pending:
            logger.info("All videos already processed!")
            self._log_cost_summary(videos)
            return videos

        # Apply max_videos limit if set
        if self.config.max_videos is not None and len(pending) > self.config.max_videos:
            logger.info(
                f"Limiting to {self.config.max_videos} videos (use --no-limit to process all {len(pending)})"
            )
            pending = pending[:self.config.max_videos]

        # Log pricing info
        audio_rate, output_rate = get_pricing(self.config.model)
        logger.info(f"Using model: {self.config.model} (${audio_rate}/1M audio, ${output_rate}/1M output)")
        logger.info(f"Processing {len(pending)} pending videos ({done_count}/{total} done)")

        # Reset running totals
        self._running_cost = 0.0
        self._running_input_tokens = 0
        self._running_output_tokens = 0
        self._videos_processed = 0

        # Create transcriber
        transcriber_config = TranscriberConfig(
            audio_dir=self.config.audio_dir,
            model=self.config.model,
        )
        transcriber = AudioTranscriber(config=transcriber_config)

        for i, video in enumerate(pending, start=1):
            if self._shutdown_requested:
                logger.info("Shutdown requested. Saving progress...")
                break

            current_num = done_count + i
            logger.info(f"\n[{current_num}/{total}] Processing: {video.title}")

            # Update status to processing
            self._update_video_status(videos, video.video_id, "processing")

            try:
                # Process video
                json_path, _, result = transcriber.process(
                    url=video.url,
                    delete_audio=self.config.delete_audio,
                )
                logger.info(f"  Saved: {json_path}")

                # Calculate and display cost
                cost = calculate_cost(self.config.model, result.input_tokens, result.output_tokens)
                logger.info(f"  Cost: ${cost:.4f} ({result.input_tokens:,} in / {result.output_tokens:,} out)")

                # Update running totals
                self._running_cost += cost
                self._running_input_tokens += result.input_tokens
                self._running_output_tokens += result.output_tokens
                self._videos_processed += 1
                logger.info(f"  Running total: ${self._running_cost:.4f} ({self._videos_processed} videos this run)")

                # Mark as done with token stats
                self._update_video_status(
                    videos, video.video_id, "done",
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                )

            except Exception as e:
                error_msg = str(e)
                logger.error(f"  Error: {error_msg}")
                self._update_video_status(videos, video.video_id, "error", error_msg)

            # Rate limiting delay between videos
            if i < len(pending) and not self._shutdown_requested:
                logger.info(f"  Waiting {self.config.delay_between_videos}s before next video...")
                time.sleep(self.config.delay_between_videos)

        # Final summary
        self._log_cost_summary(videos)

        return videos

    def _log_cost_summary(self, videos: dict[str, VideoInfo]):
        """Log final summary with cost totals."""
        done_count = sum(1 for v in videos.values() if v.status == "done")
        error_count = sum(1 for v in videos.values() if v.status == "error")
        pending_count = sum(1 for v in videos.values() if v.status not in ("done", "error"))

        total_input = sum(v.input_tokens for v in videos.values())
        total_output = sum(v.output_tokens for v in videos.values())
        total_cost = sum(v.cost_usd for v in videos.values())

        logger.info(f"\nProgress saved to: {self.config.csv_path}")
        logger.info(f"Summary: {done_count} done, {error_count} errors, {pending_count} pending")

        # This run stats
        if self._videos_processed > 0:
            logger.info(f"This run: {self._videos_processed} videos, ${self._running_cost:.4f}")
            logger.info(f"  Tokens: {self._running_input_tokens:,} in / {self._running_output_tokens:,} out")

        # All-time stats
        logger.info(f"All-time total: {total_input:,} input / {total_output:,} output tokens")
        logger.info(f"All-time estimated cost: ${total_cost:.4f}")
        logger.info("Note: Free tier users are not charged (limits: ~20 requests/day for 2.5 Flash)")


def main():
    """CLI entry point for the playlist processor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process all videos from a YouTube playlist"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--playlist",
        help="YouTube playlist URL to process",
    )
    group.add_argument(
        "--csv",
        type=Path,
        help="Resume from existing CSV (no playlist fetch)",
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
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between videos in seconds",
    )
    parser.add_argument(
        "--delete-audio",
        action="store_true",
        help="Delete audio files after transcription",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=20,
        help="Maximum number of videos to process per run (default: 20)",
    )
    parser.add_argument(
        "--no-limit",
        action="store_true",
        help="Process all pending videos (no limit)",
    )

    args = parser.parse_args()

    # Build config
    config = PlaylistConfig(
        audio_dir=args.audio_dir,
        model=args.model,
        delay_between_videos=args.delay,
        delete_audio=args.delete_audio,
        max_videos=None if args.no_limit else args.max_videos,
    )

    if args.csv:
        config.csv_path = args.csv

    processor = PlaylistProcessor(config=config)

    try:
        if args.playlist:
            processor.process_playlist(args.playlist)
        else:
            processor.process_from_csv()
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
