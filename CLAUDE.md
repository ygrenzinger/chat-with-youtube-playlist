# CLAUDE.md

RAG pipeline for YouTube playlist transcription and semantic search using Gemini AI.

## Technologies

| Library | Purpose |
|---------|---------|
| google-genai | Transcription and embeddings via Gemini API |
| yt-dlp | YouTube audio download and metadata extraction |
| sqliteai-vector | SQLite vector extension for similarity search |
| numpy | Vector operations for embeddings |
| uv | Package manager (use `uv sync` to install) |

**Requires**: Python 3.11+, `GOOGLE_API_KEY` environment variable

## Best Practices

This codebase follows these patterns:

- **Typed Python**: Union types (`str | None`), dataclasses for config and data models
- **Lazy initialization**: API clients loaded on first use (`_client: genai.Client | None = None`)
- **pathlib**: All file operations use `Path`, never string concatenation
- **Structured logging**: Module-level `logger = logging.getLogger(__name__)`
- **Error handling**: Retry logic with exponential backoff for API rate limits (429/RESOURCE_EXHAUSTED)
- **Import order**: stdlib > third-party > local
- **Context managers**: `with VectorStore() as store:` for DB connections
- **Signal handling**: Graceful shutdown on SIGINT/SIGTERM

## Scripts

| Script | Purpose | Docs |
|--------|---------|------|
| `python -m src.ingestion.playlist_processor` | Batch process YouTube playlists | [playlist_processor.md](src/ingestion/playlist_processor.md) |
| `python -m src.ingestion.audio_transcriber` | Transcribe single video | [audio_transcriber.md](src/ingestion/audio_transcriber.md) |
| `python -m src.ingestion.chunker` | Semantic chunking for RAG | [chunker.md](src/ingestion/chunker.md) |
| `python -m src.ingestion.vector_store` | Vector DB ingest/search | [vector_store.md](src/ingestion/vector_store.md) |

## Quick Reference

```bash
# Full pipeline
python -m src.ingestion.playlist_processor --playlist "URL" --max-videos 5
python -m src.ingestion.chunker --batch --input-dir data/audio/ --output-dir data/chunks/
python -m src.ingestion.vector_store ingest --input-dir data/chunks/
python -m src.ingestion.vector_store search "query" -k 5 --with-context
```
