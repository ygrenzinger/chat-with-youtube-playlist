# Component: Hybrid Time-Semantic Chunker

## Purpose

Split preprocessed transcripts into chunks optimized for retrieval:
- Semantically coherent (complete thoughts)
- Appropriately sized for BGE-M3 (~350 tokens)
- Preserving timestamp ranges for deep-linking

## Input

Processed transcript with sentences and timestamps.

## Output

List of chunks per video:

```json
{
  "video_id": "abc123xyz",
  "video_title": "Building RAG Systems - Flow Conference 2024",
  "total_chunks": 15,
  "chunks": [
    {
      "chunk_id": "abc123xyz_001",
      "chunk_index": 0,
      "text": "Bonjour à tous. Aujourd'hui, on va parler de systèmes RAG...",
      "start_time": 0.0,
      "end_time": 45.2,
      "token_count": 342,
      "section_type": "introduction",
      "context_before": "",
      "context_after": "Le premier concept important est l'embedding...",
      "youtube_link": "https://youtube.com/watch?v=abc123xyz&t=0"
    }
  ]
}
```

## Why Hybrid Chunking?

### The Problem with Pure Approaches

**Pure time-based (e.g., every 2 minutes):**
- Cuts mid-sentence or mid-thought
- Mixed topics in one chunk
- Context loss across boundaries

**Pure semantic (e.g., topic modeling):**
- Over-fragmentation (each sentence separate)
- Lost narrative flow
- No temporal anchoring
- Examples separated from explanations

### The Hybrid Solution

Uses **time as structure** and **semantics as refinement**:

1. **Coarse segmentation**: Detect major sections (intro, main, Q&A, conclusion)
2. **Fine segmentation**: Split within sections by semantic coherence
3. **Overlap**: Add context from adjacent chunks

## Functional Requirements

### 1. Section Detection (Coarse Segmentation)

Identify major section boundaries using signals:

| Signal Type | Examples | Action |
|-------------|----------|--------|
| Explicit markers | "Passons à", "Let's move to", "Première partie" | Hard boundary |
| Topic shifts | Embedding distance > threshold | Soft boundary |
| Long pauses | Gap > 3 seconds | Potential boundary |
| Q&A transition | "Des questions?", "Any questions?" | Hard boundary |
| Conclusion markers | "Pour conclure", "En résumé" | Hard boundary |

**Section types to detect:**

| Type | Description | Typical Duration |
|------|-------------|------------------|
| `introduction` | Speaker intro, agenda | 2-5 minutes |
| `main_content` | Core talk content | 15-40 minutes |
| `demo` | Live coding, demonstration | 5-15 minutes |
| `qa` | Question and answer | 5-15 minutes |
| `conclusion` | Summary, closing remarks | 2-5 minutes |

**Section detection patterns (French + English):**

```python
section_patterns = {
    "introduction": [
        r"bonjour.*(?:tous|tout le monde)",
        r"je (?:vais|voudrais) (?:me présenter|vous présenter)",
        r"(?:today|aujourd'hui).*(?:talk|parler) about",
        r"welcome to",
    ],
    "main_content": [
        r"(?:commençons|let's start|let's begin)",
        r"(?:premier|first) (?:point|topic|sujet)",
        r"(?:entrons|plongeons) dans",
    ],
    "demo": [
        r"(?:let me|je vais vous) (?:show|montrer)",
        r"(?:demo|démonstration)",
        r"(?:live|en direct)",
        r"(?:code|écran)",
    ],
    "qa": [
        r"(?:questions|des questions)",
        r"(?:q&a|q et a)",
        r"(?:floor is|la parole est)",
    ],
    "conclusion": [
        r"(?:pour (?:conclure|résumer)|to (?:conclude|summarize))",
        r"(?:en résumé|in summary)",
        r"(?:merci.*(?:attention|écoute)|thank you)",
        r"(?:wrap up|terminer)",
    ]
}
```

### 2. Chunk Formation (Fine Segmentation)

Within each section, create chunks following these rules:

**Size constraints:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Target | 350 tokens | Optimal for BGE-M3 retrieval |
| Minimum | 150 tokens | Avoid fragments |
| Maximum | 500 tokens | Hard limit for coherence |
| Flexibility | ±50 tokens | Allow natural boundaries |

**Boundary rules:**

- ✅ Always split on sentence boundaries
- ❌ Never split in the middle of:
  - An example ("Par exemple..." must stay with illustration)
  - A list ("Premièrement... Deuxièmement..." stays together if <500 tokens)
  - A code explanation (speaker describes code line by line)

**Cohesion rules:**

| Pattern | Action |
|---------|--------|
| Question + Answer (Q&A) | Keep together |
| Problem + Solution | Keep together |
| Claim + Evidence | Keep together |
| Definition + Example | Keep together |
| "If... then..." | Keep together |

### 3. Continuation Patterns

These patterns indicate the next sentence should stay with the current chunk:

```python
continuation_patterns = [
    # Examples
    r"^(Par exemple|For example|For instance|Prenons)",
    # Clarification
    r"^(C'est-à-dire|That is|In other words|Autrement dit)",
    # Consequence
    r"^(Donc|So|Thus|Therefore|Alors)",
    # Addition
    r"^(Et |And |De plus|Furthermore|Also)",
    # Contrast
    r"^(Mais|But|However|Cependant|Toutefois)",
    # Cause
    r"^(Parce que|Because|Car|Since)",
    # Reference
    r"^(Ce qui|Which|That|Cela)",
    # Lists
    r"^\d+\.",  # Numbered list continuation
    r"^(Deuxièmement|Troisièmement|Second|Third|Ensuite)",
    # Demonstration
    r"^(Ici|Here|Voici|Look|Regardez)",
]
```

### 4. Overlap Strategy

Add context overlap to preserve continuity:

| Overlap Type | Size | Purpose |
|--------------|------|---------|
| `context_before` | ~50 tokens (1-2 sentences) | Continuity from previous |
| `context_after` | ~50 tokens (1-2 sentences) | Preview of next |

**Important:** Overlap is stored separately, not duplicated in main text (avoid redundant embeddings).

```json
{
  "text": "The main content of this chunk...",
  "context_before": "[Previous chunk ending]...",
  "context_after": "...[Next chunk beginning]"
}
```

### 5. Timestamp Assignment

Each chunk must have accurate time bounds:

| Field | Source | Format |
|-------|--------|--------|
| `start_time` | First sentence start | Seconds (float) |
| `end_time` | Last sentence end | Seconds (float) |
| `youtube_link` | Computed | `https://youtube.com/watch?v={id}&t={start}` |

### 6. Metadata Enrichment

Attach to each chunk:

```json
{
  "chunk_id": "video123_005",
  "chunk_index": 4,
  "video_id": "video123",
  "video_title": "Talk Title",
  "channel": "Flow Conference",
  "section_type": "main_content",
  "language": "fr",
  "token_count": 342,
  "start_time": 180.5,
  "end_time": 225.3,
  "youtube_link": "https://youtube.com/watch?v=video123&t=180",
  "speaker": null
}
```

## Chunking Algorithm

```python
def hybrid_chunk(transcript: ProcessedTranscript, config: ChunkConfig) -> List[Chunk]:
    """
    Main chunking algorithm.
    
    1. Detect section boundaries
    2. Within sections, chunk by token count respecting boundaries
    3. Apply overlap for context preservation
    """
    
    # Step 1: Coarse segmentation - detect sections
    sections = detect_sections(transcript.sentences)
    
    all_chunks = []
    chunk_index = 0
    
    for section in sections:
        sentences = section.sentences
        
        current_chunk_sentences = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = count_tokens(sentence.text)
            
            # Check if adding this sentence exceeds target
            would_exceed = current_tokens + sentence_tokens > config.max_tokens
            
            if would_exceed and current_tokens >= config.min_tokens:
                # Check for continuation patterns before splitting
                if is_continuation(sentence) and current_tokens + sentence_tokens <= config.hard_max:
                    # Allow overflow for cohesion
                    current_chunk_sentences.append(sentence)
                    current_tokens += sentence_tokens
                    continue
                
                # Finalize current chunk
                chunk = create_chunk(
                    sentences=current_chunk_sentences,
                    index=chunk_index,
                    section=section,
                    transcript=transcript
                )
                all_chunks.append(chunk)
                chunk_index += 1
                
                # Reset for next chunk
                current_chunk_sentences = []
                current_tokens = 0
            
            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens
        
        # Handle remaining sentences in section
        if current_chunk_sentences:
            if current_tokens >= config.min_tokens:
                chunk = create_chunk(
                    sentences=current_chunk_sentences,
                    index=chunk_index,
                    section=section,
                    transcript=transcript
                )
                all_chunks.append(chunk)
                chunk_index += 1
            elif all_chunks:
                # Merge with previous chunk if too small
                all_chunks[-1] = merge_chunks(all_chunks[-1], current_chunk_sentences)
    
    # Step 3: Add overlap context
    add_overlap_context(all_chunks, config.overlap_tokens)
    
    return all_chunks


def detect_sections(sentences: List[Sentence]) -> List[Section]:
    """
    Detect section boundaries using patterns and pauses.
    """
    sections = []
    current_section = Section(type="introduction", sentences=[])
    
    for i, sentence in enumerate(sentences):
        # Check for section markers
        detected_type = detect_section_type(sentence.text)
        
        # Check for long pause (if previous sentence exists)
        long_pause = False
        if i > 0:
            gap = sentence.start - sentences[i-1].end
            long_pause = gap > 3.0  # 3 second threshold
        
        # Start new section if marker found or long pause with topic shift
        if detected_type and detected_type != current_section.type:
            if current_section.sentences:
                sections.append(current_section)
            current_section = Section(type=detected_type, sentences=[])
        
        current_section.sentences.append(sentence)
    
    # Don't forget last section
    if current_section.sentences:
        sections.append(current_section)
    
    return sections


def is_continuation(sentence: Sentence) -> bool:
    """Check if sentence should stay with previous chunk."""
    for pattern in continuation_patterns:
        if re.match(pattern, sentence.text, re.IGNORECASE):
            return True
    return False


def add_overlap_context(chunks: List[Chunk], overlap_tokens: int):
    """Add context_before and context_after to each chunk."""
    for i, chunk in enumerate(chunks):
        # Context from previous chunk
        if i > 0:
            prev_text = chunks[i-1].text
            chunk.context_before = get_last_n_tokens(prev_text, overlap_tokens)
        
        # Context from next chunk
        if i < len(chunks) - 1:
            next_text = chunks[i+1].text
            chunk.context_after = get_first_n_tokens(next_text, overlap_tokens)
```

## Configuration

```yaml
# config/chunking.yaml
chunking:
  target_tokens: 350
  min_tokens: 150
  max_tokens: 450
  hard_max_tokens: 500
  overlap_tokens: 50
  
  section_detection:
    enabled: true
    pause_threshold_seconds: 3.0
    use_embedding_similarity: false  # Can enable for better detection
    
  continuation:
    respect_examples: true
    respect_lists: true
    max_continuation_overflow: 100  # tokens
```

## Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Very short videos (<5 min) | May produce only 2-3 chunks, that's OK |
| Very long videos (>2 hours) | Process normally, will produce many chunks |
| All-demo videos | May have lower info density, allow larger chunks |
| Panel discussions | More frequent topic switches, shorter chunks |
| Rapid Q&A | Each Q+A pair should be one chunk |
| Non-talk content (trailers) | Tag as `non_content` or skip |

## Output Location

- Chunked data: `data/chunks/{video_id}_chunks.json`
- Chunk manifest: `data/chunks/manifest.json`

## Success Criteria

- [ ] Average chunk size is 300-400 tokens
- [ ] No chunks below 150 tokens (except edge cases)
- [ ] No chunks above 500 tokens
- [ ] Examples stay with their explanations
- [ ] Timestamps are accurate and link correctly
- [ ] Section detection works for most talks
- [ ] Overlap provides useful context without redundancy
