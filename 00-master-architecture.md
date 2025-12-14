# Flow Conference RAG System — Architecture Overview

## Project Goal

Build a conversational application that allows users to chat with the content from the Flow Conference YouTube playlist (https://www.youtube.com/@flowconfrance/videos). Users should be able to ask questions about talks, get summaries, find specific topics, and receive answers with direct video timestamp links.

## Technology Stack

| Component          | Technology                | Rationale                                      |
|--------------------|---------------------------|------------------------------------------------|
| Transcript Source  | YouTube auto-captions     | Available for all videos, free                 |
| Embedding Model    | BGE-M3                    | Multilingual, dense+sparse hybrid, 8192 tokens |
| Vector Store       | sqlite-vec                | Lightweight, local, no infra overhead          |
| Chat Interface     | Chainlit                  | Python-native, streaming, easy to deploy       |
| LLM (Generation)   | Claude / OpenAI / Local   | Configurable based on user preference          |

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE (Offline)                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐ │
│  │   YouTube    │───▶│  Transcript  │───▶│    Hybrid    │───▶│  BGE-M3  │ │
│  │   Playlist   │    │  Extraction  │    │   Chunking   │    │ Encoding │ │
│  └──────────────┘    │     +        │    │              │    └────┬─────┘ │
│                      │ Preprocessing│    │  (~350 tok)  │         │       │
│                      └──────────────┘    └──────────────┘         │       │
│                                                                    │       │
│                                                    ┌───────────────▼─────┐ │
│                                                    │    sqlite-vec       │ │
│                                                    │  (vectors + meta)   │ │
│                                                    └───────────────┬─────┘ │
└────────────────────────────────────────────────────────────────────┼───────┘
                                                                     │
┌────────────────────────────────────────────────────────────────────┼───────┐
│                         QUERY PIPELINE (Online)                    │       │
├────────────────────────────────────────────────────────────────────┼───────┤
│                                                                    │       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │       │
│  │   Chainlit   │───▶│   BGE-M3     │───▶│  sqlite-vec  │◀────────┘       │
│  │  User Query  │    │   Encode     │    │  Retrieval   │                 │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                 │
│                                                  │                         │
│                                                  ▼                         │
│                                          ┌──────────────┐                 │
│                                          │   Reranking  │                 │
│                                          │  (optional)  │                 │
│                                          └──────┬───────┘                 │
│                                                  │                         │
│                                                  ▼                         │
│  ┌──────────────┐    ┌──────────────────────────────────┐                 │
│  │   Chainlit   │◀───│   LLM Generation + Citations     │                 │
│  │   Response   │    │   (with timestamp deep links)    │                 │
│  └──────────────┘    └──────────────────────────────────┘                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

### 1. Ingestion (run once per new video)

1. Fetch video list from YouTube channel
2. Extract auto-generated transcripts with timestamps
3. Preprocess: punctuation restoration, cleaning, sentence merging
4. Chunk using hybrid time+semantic strategy (~350 tokens, 15% overlap)
5. Embed chunks with BGE-M3 (dense vectors, optionally sparse)
6. Store in sqlite-vec with full metadata

### 2. Query (real-time)

1. User asks question in Chainlit interface
2. Query embedded with BGE-M3
3. Retrieve top-k similar chunks from sqlite-vec
4. Optionally rerank results
5. Generate response with LLM, including source citations
6. Display response with clickable timestamp links

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **BGE-M3** | Supports French + English (Flow Conference has both), handles long context, provides dense+sparse for hybrid search |
| **sqlite-vec** | Zero infrastructure, single file DB, easy backup/versioning |
| **Hybrid chunking** | Preserves talk structure, keeps examples with explanations |
| **Timestamp preservation** | Every chunk links back to exact video moment |

## File Structure

```
flow-conference-rag/
├── src/
│   ├── ingestion/
│   │   ├── youtube_extractor.py
│   │   ├── preprocessor.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   └── reranker.py
│   ├── generation/
│   │   └── llm_chain.py
│   └── app.py                 # Chainlit application
├── data/
│   ├── raw_transcripts/
│   ├── processed_chunks/
│   └── flow_conference.db     # sqlite-vec database
├── config/
│   └── settings.yaml
├── chainlit.md
└── requirements.txt
```

## Non-Functional Requirements

- Support incremental ingestion (add new videos without reprocessing all)
- Response latency < 3 seconds for retrieval + generation
- Work fully offline after initial setup (except LLM API calls if using cloud)
- Handle both French and English content seamlessly

## Component Specifications

Each component has a detailed specification document:

| Document | Description |
|----------|-------------|
| `01-transcript-extractor.md` | YouTube transcript extraction |
| `02-preprocessor.md` | Transcript cleaning and preprocessing |
| `03-chunker.md` | Hybrid time-semantic chunking |
| `04-embedder.md` | BGE-M3 embedding generation |
| `05-vector-store.md` | sqlite-vec storage and retrieval |
| `06-retrieval-generation.md` | Query pipeline and LLM generation |
| `07-chainlit-app.md` | Chat interface application |
