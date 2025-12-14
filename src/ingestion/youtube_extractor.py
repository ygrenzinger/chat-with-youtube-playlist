"""
YouTube Transcript Extractor using yt-dlp.

Extracts auto-generated and manual transcripts from all videos in a YouTube channel,
preserving timestamps for deep-linking.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yt_dlp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class YouTubeExtractor:
    """Extract transcripts from YouTube channel videos using yt-dlp."""

    LANGUAGE_PRIORITY = ["fr", "en"]
    DEFAULT_CHANNEL_URL = "https://www.youtube.com/@flowconfrance/videos"
    REQUEST_DELAY = 1.5  # seconds between requests to avoid throttling

    def __init__(self, output_dir: str = "data/raw_transcripts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Load or create processing manifest for incremental updates."""
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {"processed_videos": {}, "last_run": None, "errors": {}}

    def _save_manifest(self):
        """Save processing manifest."""
        self.manifest["last_run"] = datetime.now().isoformat()
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def _get_yt_dlp_options(self, for_subtitles: bool = False) -> dict:
        """Get yt-dlp options for different operations."""
        base_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": not for_subtitles,
        }

        if for_subtitles:
            base_opts.update({
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": self.LANGUAGE_PRIORITY,
                "skip_download": True,
                "extract_flat": False,
            })

        return base_opts

    def get_channel_videos(self, channel_url: str) -> list[dict[str, Any]]:
        """
        Fetch all video metadata from a YouTube channel.

        Args:
            channel_url: URL to the YouTube channel videos page

        Returns:
            List of video metadata dictionaries
        """
        logger.info(f"Fetching video list from channel: {channel_url}")

        opts = self._get_yt_dlp_options(for_subtitles=False)

        with yt_dlp.YoutubeDL(opts) as ydl:
            result = ydl.extract_info(channel_url, download=False)

        if not result or "entries" not in result:
            logger.warning("No videos found in channel")
            return []

        videos = []
        for entry in result["entries"]:
            if entry is None:
                continue

            videos.append({
                "video_id": entry.get("id"),
                "title": entry.get("title"),
                "url": entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}",
                "duration": entry.get("duration"),
                "upload_date": entry.get("upload_date"),
            })

        logger.info(f"Found {len(videos)} videos in channel")
        return videos

    def get_transcript(self, video_id: str) -> dict[str, Any]:
        """
        Fetch transcript for a single video with language priority.

        Prefers manually uploaded subtitles over auto-generated,
        and French over English.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary with transcript data or error information
        """
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"Fetching transcript for video: {video_id}")

        opts = {
            "quiet": True,
            "no_warnings": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": self.LANGUAGE_PRIORITY,
            "skip_download": True,
        }

        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=False)

            if not info:
                return {"error": "Could not extract video info"}

            # Get available subtitles
            manual_subs = info.get("subtitles", {})
            auto_subs = info.get("automatic_captions", {})

            # Try to find the best subtitle track (manual preferred, then auto)
            selected_sub = None
            sub_type = None
            selected_lang = None

            # First try manual subtitles in language priority order
            for lang in self.LANGUAGE_PRIORITY:
                if lang in manual_subs and manual_subs[lang]:
                    selected_sub = manual_subs[lang]
                    sub_type = "manual"
                    selected_lang = lang
                    break

            # Fall back to auto-generated
            if not selected_sub:
                for lang in self.LANGUAGE_PRIORITY:
                    if lang in auto_subs and auto_subs[lang]:
                        selected_sub = auto_subs[lang]
                        sub_type = "auto"
                        selected_lang = lang
                        break

            if not selected_sub:
                return {"error": "No transcript available in supported languages"}

            # Find JSON3 format for structured timestamps, fall back to others
            sub_url = None
            for fmt in selected_sub:
                if fmt.get("ext") == "json3":
                    sub_url = fmt.get("url")
                    break

            if not sub_url:
                # Try srv3 or vtt format
                for fmt in selected_sub:
                    if fmt.get("ext") in ["srv3", "vtt", "ttml"]:
                        sub_url = fmt.get("url")
                        break

            if not sub_url and selected_sub:
                sub_url = selected_sub[0].get("url")

            if not sub_url:
                return {"error": "Could not find subtitle URL"}

            # Download and parse the subtitle content
            transcript_data = self._download_and_parse_subtitles(sub_url, video_id)

            if "error" in transcript_data:
                return transcript_data

            return {
                "video_id": video_id,
                "title": info.get("title"),
                "channel": info.get("channel") or info.get("uploader"),
                "upload_date": info.get("upload_date"),
                "duration_seconds": info.get("duration"),
                "language": selected_lang,
                "transcript_type": sub_type,
                "transcript": transcript_data["transcript"],
            }

        except Exception as e:
            logger.error(f"Error fetching transcript for {video_id}: {e}")
            return {"error": str(e)}

    def _download_and_parse_subtitles(self, sub_url: str, video_id: str) -> dict[str, Any]:
        """Download subtitle file and parse into structured format."""
        import urllib.request

        try:
            with urllib.request.urlopen(sub_url, timeout=30) as response:
                content = response.read().decode("utf-8")

            # Try to parse as JSON3 (YouTube's native format)
            if sub_url.endswith("json3") or "fmt=json3" in sub_url:
                return self._parse_json3_subtitles(content)

            # Try to parse as srv3/ttml
            if "srv3" in sub_url or sub_url.endswith(".ttml"):
                return self._parse_srv3_subtitles(content)

            # Fall back to VTT parsing
            return self._parse_vtt_subtitles(content)

        except Exception as e:
            logger.error(f"Error downloading subtitles for {video_id}: {e}")
            return {"error": f"Failed to download subtitles: {e}"}

    def _parse_json3_subtitles(self, content: str) -> dict[str, Any]:
        """Parse JSON3 format subtitles from YouTube."""
        try:
            data = json.loads(content)
            transcript = []

            events = data.get("events", [])
            for event in events:
                # Skip events without segments (usually metadata)
                if "segs" not in event:
                    continue

                start_ms = event.get("tStartMs", 0)
                duration_ms = event.get("dDurationMs", 0)

                # Combine all segments in this event
                text_parts = []
                for seg in event.get("segs", []):
                    text = seg.get("utf8", "")
                    if text and text.strip():
                        text_parts.append(text)

                combined_text = "".join(text_parts).strip()
                if combined_text:
                    transcript.append({
                        "start": start_ms / 1000.0,
                        "duration": duration_ms / 1000.0,
                        "text": combined_text,
                    })

            return {"transcript": transcript}

        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON3 subtitles: {e}"}

    def _parse_srv3_subtitles(self, content: str) -> dict[str, Any]:
        """Parse srv3/TTML format subtitles."""
        import re

        transcript = []

        # Simple regex to extract text and timing from TTML/srv3
        pattern = r'<p[^>]*begin="([^"]+)"[^>]*end="([^"]+)"[^>]*>(.*?)</p>'
        matches = re.findall(pattern, content, re.DOTALL)

        for begin, end, text in matches:
            start_sec = self._parse_timestamp(begin)
            end_sec = self._parse_timestamp(end)

            # Clean HTML tags from text
            clean_text = re.sub(r"<[^>]+>", "", text).strip()

            if clean_text:
                transcript.append({
                    "start": start_sec,
                    "duration": end_sec - start_sec,
                    "text": clean_text,
                })

        return {"transcript": transcript}

    def _parse_vtt_subtitles(self, content: str) -> dict[str, Any]:
        """Parse VTT format subtitles."""
        import re

        transcript = []
        lines = content.split("\n")

        timestamp_pattern = r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})"

        i = 0
        while i < len(lines):
            match = re.match(timestamp_pattern, lines[i])
            if match:
                start_time = self._parse_vtt_timestamp(match.group(1))
                end_time = self._parse_vtt_timestamp(match.group(2))

                # Collect text lines until empty line or next timestamp
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip() and not re.match(timestamp_pattern, lines[i]):
                    # Remove VTT formatting tags
                    clean_line = re.sub(r"<[^>]+>", "", lines[i]).strip()
                    if clean_line:
                        text_lines.append(clean_line)
                    i += 1

                if text_lines:
                    transcript.append({
                        "start": start_time,
                        "duration": end_time - start_time,
                        "text": " ".join(text_lines),
                    })
            else:
                i += 1

        return {"transcript": transcript}

    def _parse_timestamp(self, timestamp: str) -> float:
        """Parse various timestamp formats to seconds."""
        import re

        # Handle HH:MM:SS.mmm or MM:SS.mmm
        if ":" in timestamp:
            parts = timestamp.replace(",", ".").split(":")
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)

        # Handle seconds with optional milliseconds
        match = re.match(r"(\d+\.?\d*)s?", timestamp)
        if match:
            return float(match.group(1))

        return 0.0

    def _parse_vtt_timestamp(self, timestamp: str) -> float:
        """Parse VTT timestamp format (HH:MM:SS.mmm)."""
        parts = timestamp.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        return 0.0

    def process_video(self, video_id: str, title: str | None = None) -> bool:
        """
        Process a single video and save its transcript.

        Args:
            video_id: YouTube video ID
            title: Optional video title (will be fetched if not provided)

        Returns:
            True if successful, False otherwise
        """
        result = self.get_transcript(video_id)

        if "error" in result:
            logger.warning(f"Error processing {video_id}: {result['error']}")
            self.manifest["errors"][video_id] = {
                "error": result["error"],
                "timestamp": datetime.now().isoformat(),
            }
            self._save_manifest()
            return False

        # Save transcript
        output_path = self.output_dir / f"{video_id}.json"
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

        # Update manifest
        self.manifest["processed_videos"][video_id] = {
            "processed_at": datetime.now().isoformat(),
            "title": result.get("title"),
            "language": result.get("language"),
            "transcript_type": result.get("transcript_type"),
            "segment_count": len(result.get("transcript", [])),
        }

        # Remove from errors if previously failed
        if video_id in self.manifest["errors"]:
            del self.manifest["errors"][video_id]

        self._save_manifest()
        logger.info(f"Saved transcript for {video_id}: {result.get('title')}")
        return True

    def process_channel(self, channel_url: str | None = None, force_reprocess: bool = False):
        """
        Process all videos from a YouTube channel.

        Args:
            channel_url: URL to the channel videos page (defaults to Flow Conference)
            force_reprocess: If True, reprocess already-processed videos
        """
        channel_url = channel_url or self.DEFAULT_CHANNEL_URL
        videos = self.get_channel_videos(channel_url)

        if not videos:
            logger.warning("No videos to process")
            return

        # Filter out already processed videos unless force_reprocess
        if not force_reprocess:
            videos = [
                v for v in videos
                if v["video_id"] not in self.manifest["processed_videos"]
            ]
            logger.info(f"{len(videos)} new videos to process")

        success_count = 0
        error_count = 0

        for i, video in enumerate(videos):
            video_id = video["video_id"]
            logger.info(f"Processing video {i + 1}/{len(videos)}: {video.get('title', video_id)}")

            if self.process_video(video_id, video.get("title")):
                success_count += 1
            else:
                error_count += 1

            # Rate limiting
            if i < len(videos) - 1:
                time.sleep(self.REQUEST_DELAY)

        logger.info(f"Processing complete. Success: {success_count}, Errors: {error_count}")

    def retry_failed(self):
        """Retry processing videos that previously failed."""
        failed_ids = list(self.manifest.get("errors", {}).keys())

        if not failed_ids:
            logger.info("No failed videos to retry")
            return

        logger.info(f"Retrying {len(failed_ids)} failed videos")

        for i, video_id in enumerate(failed_ids):
            logger.info(f"Retrying {i + 1}/{len(failed_ids)}: {video_id}")
            self.process_video(video_id)

            if i < len(failed_ids) - 1:
                time.sleep(self.REQUEST_DELAY)


def main():
    """CLI entry point for the YouTube extractor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract transcripts from YouTube channel videos"
    )
    parser.add_argument(
        "--channel",
        default=YouTubeExtractor.DEFAULT_CHANNEL_URL,
        help="YouTube channel URL to process",
    )
    parser.add_argument(
        "--output",
        default="data/raw_transcripts",
        help="Output directory for transcripts",
    )
    parser.add_argument(
        "--video-id",
        help="Process a single video by ID",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of already-processed videos",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry processing of previously failed videos",
    )

    args = parser.parse_args()

    extractor = YouTubeExtractor(output_dir=args.output)

    if args.video_id:
        extractor.process_video(args.video_id)
    elif args.retry_failed:
        extractor.retry_failed()
    else:
        extractor.process_channel(args.channel, force_reprocess=args.force)


if __name__ == "__main__":
    main()
