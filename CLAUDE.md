# Flow Conference RAG System

Conversational app to chat with Flow Conference YouTube playlist content. Users ask questions and get answers with direct video timestamp links.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Package Manager | uv |
| Embedding | BGE-M3 (multilingual, 1024 dim, 8192 tokens) |
| Vector Store | sqlite-vec |
| Chat UI | Chainlit |
| LLM | Gemini 2.5 Flash (default), Claude / OpenAI configurable |
| Transcripts | yt-dlp |

## Project Structure

```
src/
├── ingestion/
│   ├── youtube_extractor.py  # Fetch transcripts from YouTube
│   ├── preprocessor.py       # Clean and normalize text
│   ├── chunker.py            # Hybrid time+semantic chunking (~350 tokens)
│   └── embedder.py           # BGE-M3 embedding generation
├── retrieval/
│   ├── vector_store.py       # sqlite-vec operations
│   └── reranker.py           # Optional reranking
├── generation/
│   └── llm_chain.py          # LLM integration with citations
└── app.py                    # Chainlit application

data/
├── raw_transcripts/          # Original YouTube transcripts
├── processed_chunks/         # Chunked and preprocessed data
└── flow_conference.db        # sqlite-vec database

config/
└── settings.yaml
```

## Setup

```bash
# Create project and install dependencies
uv init
uv add yt-dlp FlagEmbedding sqlite-vec chainlit google-genai torch numpy tqdm pyyaml python-dotenv

# Environment variables
cp .env.example .env
# Set GOOGLE_API_KEY (default), or ANTHROPIC_API_KEY / OPENAI_API_KEY
```

## Commands

```bash
# Run ingestion pipeline
uv run python -m src.ingestion.youtube_extractor

# Start chat interface
uv run chainlit run src/app.py

# Development mode with hot reload
uv run chainlit run src/app.py --watch
```

## Architecture Specs

Detailed specs in numbered markdown files:
- `00-master-architecture.md` - System overview
- `01-transcript-extractor.md` - YouTube extraction
- `02-preprocessor.md` - Text cleaning
- `03-chunker.md` - Chunking strategy
- `04-embedder.md` - BGE-M3 embeddings
- `05-vector-store.md` - sqlite-vec schema
- `06-retrieval-generation.md` - Query pipeline
- `07-chainlit-app.md` - Chat interface

## Key Design Decisions

- **BGE-M3**: Handles French + English, long context, dense+sparse hybrid search
- **sqlite-vec**: Zero infrastructure, single file DB, easy backup
- **Hybrid chunking**: ~350 tokens with 15% overlap, preserves talk structure
- **Timestamp preservation**: Every chunk links to exact video moment

## Data Flow

1. **Ingestion**: YouTube channel -> Transcripts -> Chunks -> Embeddings -> sqlite-vec
2. **Query**: User question -> Embed -> Vector search -> Rerank -> LLM generate -> Response with citations
