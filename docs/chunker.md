# Hybrid Chunker for YouTube Transcripts

A semantic chunking system optimized for RAG (Retrieval-Augmented Generation) pipelines. It processes YouTube video transcriptions and creates hierarchical chunks with parent-child relationships for improved retrieval quality.

## Architecture Overview

```mermaid
flowchart TB
    subgraph Input
        T[Gemini Transcript JSON]
    end

    subgraph Processing
        T --> S[Parse Segments]
        S --> BD[Boundary Detection]
        BD --> CC[Create Child Chunks]
        CC --> OC[Add Overlap Context]
        OC --> GP[Group into Parents]
    end

    subgraph Output
        GP --> CV[ChunkedVideo]
        CV --> CJ[chunks.json]
    end

    subgraph "Boundary Detection"
        BD --> E[Embedding Similarity]
        BD --> P[Pause Detection]
        E --> |cosine < 0.5| B[Boundaries]
        P --> |gap > 3s| B
    end
```

## Parent-Child Hierarchy

The chunker implements a two-level hierarchy optimized for RAG:

```mermaid
flowchart TB
    subgraph "Parent Chunk (800 tokens)"
        P[VIDEO_P000<br/>NOT embedded<br/>Used for LLM context]
    end

    subgraph "Child Chunks (200 tokens each)"
        C1[VIDEO_000<br/>Embedded<br/>Used for search]
        C2[VIDEO_001<br/>Embedded<br/>Used for search]
        C3[VIDEO_002<br/>Embedded<br/>Used for search]
        C4[VIDEO_003<br/>Embedded<br/>Used for search]
    end

    P --> C1
    P --> C2
    P --> C3
    P --> C4
```

| Layer | Token Range | Purpose | Embedded? |
|-------|-------------|---------|-----------|
| **Parent** | 512-1000 | LLM context expansion | No |
| **Child** | 128-256 | Vector search | Yes |

## RAG Retrieval Workflow

```mermaid
sequenceDiagram
    participant U as User Query
    participant VS as Vector Store
    participant C as Child Chunks
    participant P as Parent Chunks
    participant LLM as LLM

    U->>VS: Search query embedding
    VS->>C: Find similar child chunks
    C->>P: Expand to parent chunks
    P->>LLM: Send parent text as context
    LLM->>U: Generate answer
    Note over C: Return youtube_link<br/>for citation
```

## Data Flow

```mermaid
flowchart LR
    subgraph "1. Input"
        JSON["transcript.json<br/>{segments: [...]}"]
    end

    subgraph "2. Parse"
        SEG["Segment[]<br/>text, start, end"]
    end

    subgraph "3. Detect Boundaries"
        EMB["Gemini Embeddings<br/>768 dimensions"]
        COS["Cosine Similarity<br/>< 0.5 = boundary"]
        PAUSE["Pause Detection<br/>> 3s gap"]
    end

    subgraph "4. Create Chunks"
        CHILD["Child Chunks<br/>128-256 tokens"]
        PARENT["Parent Chunks<br/>512-1000 tokens"]
    end

    subgraph "5. Output"
        OUT["ChunkedVideo<br/>chunks + parent_chunks"]
    end

    JSON --> SEG
    SEG --> EMB
    EMB --> COS
    SEG --> PAUSE
    COS --> CHILD
    PAUSE --> CHILD
    CHILD --> PARENT
    PARENT --> OUT
```

## Core Components

### Data Classes

```mermaid
classDiagram
    class ChunkConfig {
        +int child_target_tokens = 200
        +int child_min_tokens = 128
        +int child_max_tokens = 256
        +int parent_target_tokens = 800
        +int parent_min_tokens = 512
        +int parent_max_tokens = 1000
        +bool enable_parent_chunks = true
        +float similarity_threshold = 0.5
    }

    class Segment {
        +str text
        +float start
        +float end
        +str speaker
        +str language
    }

    class Chunk {
        +str chunk_id
        +str text
        +float start_time
        +float end_time
        +int token_count
        +str youtube_link
        +str parent_chunk_id
        +str context_before
        +str context_after
    }

    class ParentChunk {
        +str parent_chunk_id
        +str text
        +float start_time
        +float end_time
        +int token_count
        +list~str~ child_chunk_ids
    }

    class ChunkedVideo {
        +str video_id
        +str video_title
        +int total_chunks
        +list~Chunk~ chunks
        +int total_parent_chunks
        +list~ParentChunk~ parent_chunks
    }

    ChunkedVideo "1" --> "*" Chunk
    ChunkedVideo "1" --> "*" ParentChunk
    ParentChunk "1" --> "*" Chunk : contains
```

### HybridChunker Class

```mermaid
classDiagram
    class HybridChunker {
        -ChunkConfig config
        -GeminiEmbedding _embedding_client
        -list~Pattern~ _compiled_continuation_patterns
        +load_transcript(path) list~Segment~
        +detect_boundaries(segments) list~int~
        +chunk_transcript(segments, video_id) ChunkedVideo
        +group_into_parents(chunks, video_id) list~ParentChunk~
        +process_file(input_path, output_path) ChunkedVideo
    }

    class GeminiEmbedding {
        -str model
        -int dimensions
        -str task_type
        +embed(texts) ndarray
    }

    HybridChunker --> GeminiEmbedding
    HybridChunker --> ChunkConfig
```

## Boundary Detection

The chunker uses two complementary signals to detect natural chunk boundaries:

### 1. Semantic Boundaries (Embedding-based)

```mermaid
flowchart LR
    subgraph "Consecutive Segments"
        S1["Segment i-1"]
        S2["Segment i"]
    end

    subgraph "Embeddings"
        E1["embedding[i-1]"]
        E2["embedding[i]"]
    end

    subgraph "Similarity"
        COS["cosine_similarity()"]
        TH{"< 0.5?"}
    end

    S1 --> E1
    S2 --> E2
    E1 --> COS
    E2 --> COS
    COS --> TH
    TH -->|Yes| B["Boundary"]
    TH -->|No| NB["No Boundary"]
```

### 2. Temporal Boundaries (Pause-based)

```mermaid
flowchart LR
    subgraph "Timestamps"
        END["segment[i-1].end"]
        START["segment[i].start"]
    end

    subgraph "Gap Calculation"
        GAP["gap = start - end"]
        TH{"gap > 3s?"}
    end

    END --> GAP
    START --> GAP
    GAP --> TH
    TH -->|Yes| B["Boundary"]
    TH -->|No| NB["No Boundary"]
```

## Chunking Algorithm

```mermaid
flowchart TD
    START([Start]) --> LOAD[Load transcript segments]
    LOAD --> DETECT[Detect boundaries<br/>embeddings + pauses]
    DETECT --> INIT[Initialize: chunks=[], tokens=0]

    INIT --> LOOP{For each segment}

    LOOP --> CHECK{Would exceed<br/>max tokens?}

    CHECK -->|Yes| MIN{tokens >= min?}
    MIN -->|Yes| CONT{Is continuation<br/>pattern?}
    CONT -->|Yes| HARD{tokens <= hard_max?}
    HARD -->|Yes| ADD[Add to current chunk]
    HARD -->|No| SPLIT[Create chunk, reset]
    CONT -->|No| SPLIT
    MIN -->|No| ADD

    CHECK -->|No| TARGET{tokens >= target<br/>AND at boundary?}
    TARGET -->|Yes| SPLIT
    TARGET -->|No| ADD

    ADD --> LOOP
    SPLIT --> LOOP

    LOOP -->|Done| OVERLAP[Add overlap context]
    OVERLAP --> PARENT{Parent chunks<br/>enabled?}
    PARENT -->|Yes| GROUP[Group into parents]
    PARENT -->|No| OUTPUT
    GROUP --> OUTPUT[Return ChunkedVideo]
    OUTPUT --> END([End])
```

## Continuation Patterns

The chunker recognizes linguistic patterns that indicate a segment should stay with the previous chunk:

| Category | Patterns (EN/FR) |
|----------|-----------------|
| Examples | "For example", "Par exemple", "For instance" |
| Clarification | "That is", "C'est-à-dire", "In other words" |
| Consequence | "So", "Donc", "Therefore", "Thus" |
| Addition | "And", "Et", "Furthermore", "De plus" |
| Contrast | "But", "Mais", "However", "Cependant" |
| Reference | "This", "Cette", "Which", "Ce qui" |
| Lists | "Second", "Deuxièmement", "1.", "2." |

## Output JSON Structure

```json
{
  "video_id": "abc123",
  "video_title": "Introduction to RAG",
  "total_chunks": 20,
  "total_parent_chunks": 5,
  "parent_chunks": [
    {
      "parent_chunk_id": "abc123_P000",
      "parent_index": 0,
      "text": "Combined text from all children (~800 tokens)...",
      "start_time": 0.0,
      "end_time": 120.5,
      "token_count": 812,
      "child_chunk_ids": ["abc123_000", "abc123_001", "abc123_002", "abc123_003"]
    }
  ],
  "chunks": [
    {
      "chunk_id": "abc123_000",
      "chunk_index": 0,
      "text": "Individual chunk text (~200 tokens)...",
      "start_time": 0.0,
      "end_time": 30.2,
      "token_count": 195,
      "youtube_link": "https://youtube.com/watch?v=abc123&t=0",
      "parent_chunk_id": "abc123_P000",
      "context_before": "",
      "context_after": "...last 50 tokens of next chunk..."
    }
  ]
}
```

## CLI Usage

```bash
# Process single file (with parent chunks - default)
uv run python -m src.ingestion.chunker \
  --input data/audio/video.json \
  --output-dir data/chunks/

# Process single file (without parent chunks)
uv run python -m src.ingestion.chunker \
  --input data/audio/video.json \
  --no-parent-chunks

# Batch process all transcripts
uv run python -m src.ingestion.chunker \
  --batch \
  --input-dir data/audio/ \
  --output-dir data/chunks/

# Custom token settings
uv run python -m src.ingestion.chunker \
  --input data/audio/video.json \
  --child-target-tokens 150 \
  --child-max-tokens 200 \
  --parent-target-tokens 600 \
  --parent-max-tokens 800
```

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `child_target_tokens` | 200 | Target size for child chunks |
| `child_min_tokens` | 128 | Minimum child chunk size |
| `child_max_tokens` | 256 | Maximum child chunk size |
| `child_hard_max_tokens` | 300 | Hard max (for continuations) |
| `parent_target_tokens` | 800 | Target size for parent chunks |
| `parent_min_tokens` | 512 | Minimum parent chunk size |
| `parent_max_tokens` | 1000 | Maximum parent chunk size |
| `enable_parent_chunks` | true | Enable parent chunk generation |
| `overlap_tokens` | 50 | Context overlap between chunks |
| `pause_threshold_seconds` | 3.0 | Gap duration for pause boundary |
| `similarity_threshold` | 0.5 | Cosine similarity for boundary |
| `embedding_model` | gemini-embedding-001 | Embedding model |
| `embedding_dimensions` | 768 | Embedding vector dimensions |

## When to Use Parent Chunks

```mermaid
flowchart TD
    Q{What type of content?}

    Q -->|Complex explanations| YES[Use Parent Chunks]
    Q -->|Why/How questions| YES
    Q -->|Technical dependencies| YES

    Q -->|Self-contained FAQs| NO[Skip Parent Chunks]
    Q -->|Simple definitions| NO
    Q -->|Snippet-only retrieval| NO

    YES --> R1[Better reasoning context]
    NO --> R2[Lower storage/complexity]
```

## Dependencies

- `google-genai` - Gemini API for embeddings
- `numpy` - Vector operations
- Python 3.11+
