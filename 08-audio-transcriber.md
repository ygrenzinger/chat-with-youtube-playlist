# 08 - Audio Transcriber

Transcribes YouTube videos without subtitles using yt-dlp + Gemini.

## When to Use

Use when YouTube auto-captions are unavailable or poor quality. Falls back from `youtube_extractor.py`.

## Pipeline

```
YouTube URL → yt-dlp (audio download) → Gemini 2.5 Flash → .txt transcript
```

## Usage

```bash
# Basic usage
uv run python -m src.ingestion.audio_transcriber --url "https://youtube.com/watch?v=..."

# Custom output
uv run python -m src.ingestion.audio_transcriber --url "..." --output data/transcripts/custom.txt

# Clean up audio after transcription
uv run python -m src.ingestion.audio_transcriber --url "..." --delete-audio
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | required | YouTube video URL |
| `--output` | auto | Output path (default: `data/transcripts/{title}.txt`) |
| `--delete-audio` | false | Remove audio file after transcription |
| `--audio-dir` | `data/audio` | Audio download directory |
| `--transcript-dir` | `data/transcripts` | Transcript output directory |
| `--model` | `gemini-2.5-flash` | Gemini model for transcription |

## Configuration

```python
TranscriberConfig(
    audio_dir=Path("data/audio"),
    transcript_dir=Path("data/transcripts"),
    audio_quality="192",
    model="gemini-2.5-flash",
    retries=3,
    convert_to_mp3=False  # Gemini handles m4a directly
)
```

## Requirements

- `GOOGLE_API_KEY` environment variable
- yt-dlp, google-genai packages

## Output Format

Plain text paragraphs, no timestamps. Original language preserved (no translation).
