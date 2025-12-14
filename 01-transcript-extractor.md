# Component: YouTube Transcript Extractor

## Purpose

Extract auto-generated transcripts from all videos in the Flow Conference YouTube channel, preserving timestamps for later deep-linking.

## Input

- YouTube channel URL: `https://www.youtube.com/@flowconfrance/videos`
- Or: List of specific video IDs

## Output

For each video, produce a JSON file:

```json
{
  "video_id": "abc123xyz",
  "title": "Building RAG Systems - Flow Conference 2024",
  "channel": "Flow Conference",
  "upload_date": "2024-03-15",
  "duration_seconds": 1847,
  "language": "fr",
  "transcript": [
    {
      "start": 0.0,
      "duration": 2.4,
      "text": "bonjour à tous"
    },
    {
      "start": 2.4,
      "duration": 3.1,
      "text": "aujourd'hui on va parler de"
    }
  ]
}
```

## Functional Requirements

### 1. Channel Crawling

- Fetch all public video IDs from the channel
- Support pagination (channel may have 100+ videos)
- Extract video metadata: title, upload date, duration, description

### 2. Transcript Retrieval

- Prefer manually uploaded subtitles if available
- Fall back to auto-generated captions
- Support multiple languages (French primary, English secondary)
- Handle videos with no captions gracefully (log warning, skip)

### 3. Timestamp Preservation

- Keep original start time and duration for each segment
- Timestamps must be accurate to support deep-linking

### 4. Incremental Updates

- Track already-processed videos (by video_id)
- Only fetch new videos on subsequent runs
- Store processing state in a manifest file

### 5. Error Handling

- Retry failed requests with exponential backoff
- Log errors with video_id for manual review
- Continue processing other videos if one fails

## Technical Constraints

- Use `yt-dlp` for channel video listing and transcript fetching
- Rate limit requests to avoid YouTube throttling (1-2 req/sec)
- Store raw transcripts before any processing (reproducibility)

## Edge Cases to Handle

| Edge Case | Handling Strategy |
|-----------|-------------------|
| Videos with only auto-generated captions | Process with lower quality flag |
| Videos with captions in unexpected languages | Detect and tag language |
| Live streams | May have different caption format |
| Premiere videos | May not have captions immediately |
| Private or deleted videos | Skip with warning |
| Very long videos (3+ hours) | Process in segments if needed |

## Dependencies

```python
# requirements.txt
yt-dlp>=2025.12.8
```

## Example Implementation Skeleton

```python
from youtube_transcript_api import YouTubeTranscriptApi
import scrapetube
import json
from pathlib import Path
from datetime import datetime

class YouTubeExtractor:
    def __init__(self, output_dir: str = "data/raw_transcripts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> dict:
        """Load or create processing manifest."""
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {"processed_videos": {}, "last_run": None}
    
    def _save_manifest(self):
        """Save processing manifest."""
        self.manifest["last_run"] = datetime.now().isoformat()
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))
    
    def get_channel_videos(self, channel_url: str) -> list:
        """Fetch all video IDs from channel."""
        videos = scrapetube.get_channel(channel_url=channel_url)
        return [
            {
                "video_id": v["videoId"],
                "title": v["title"]["runs"][0]["text"],
                # ... other metadata
            }
            for v in videos
        ]
    
    def get_transcript(self, video_id: str) -> dict:
        """Fetch transcript for a single video."""
        try:
            # Try French first, then English, then auto-generated
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Priority: manual French > manual English > auto French > auto English
            for lang in ['fr', 'en']:
                try:
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    return {"transcript": transcript.fetch(), "type": "manual", "language": lang}
                except:
                    pass
            
            for lang in ['fr', 'en']:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    return {"transcript": transcript.fetch(), "type": "auto", "language": lang}
                except:
                    pass
            
            raise Exception("No transcript found")
            
        except Exception as e:
            return {"error": str(e)}
    
    def process_channel(self, channel_url: str):
        """Process all videos from channel."""
        videos = self.get_channel_videos(channel_url)
        
        for video in videos:
            video_id = video["video_id"]
            
            # Skip if already processed
            if video_id in self.manifest["processed_videos"]:
                continue
            
            # Fetch transcript
            result = self.get_transcript(video_id)
            
            if "error" in result:
                print(f"Error processing {video_id}: {result['error']}")
                continue
            
            # Save transcript
            output = {
                "video_id": video_id,
                "title": video["title"],
                "channel": "Flow Conference",
                "language": result["language"],
                "transcript_type": result["type"],
                "transcript": result["transcript"]
            }
            
            output_path = self.output_dir / f"{video_id}.json"
            output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
            
            # Update manifest
            self.manifest["processed_videos"][video_id] = {
                "processed_at": datetime.now().isoformat(),
                "status": "success"
            }
            self._save_manifest()
```

## Output Location

- Raw transcripts: `data/raw_transcripts/{video_id}.json`
- Manifest: `data/raw_transcripts/manifest.json`

## Success Criteria

- [ ] All public videos from channel are discovered
- [ ] Transcripts are fetched with correct timestamps
- [ ] Incremental updates work correctly
- [ ] Errors are logged but don't stop processing
- [ ] Rate limiting prevents YouTube blocks
