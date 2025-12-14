# Component: Retrieval and Generation Pipeline

## Purpose

Orchestrate the query flow: embed user question, retrieve relevant chunks, rerank results, and generate a cited response with video timestamp links.

## Input

User query string from Chainlit interface.

## Output

Generated response with:
- Natural language answer
- Citations to source chunks
- Clickable timestamp links to YouTube videos

## Pipeline Overview

```
User Query
    │
    ▼
┌─────────────────────────┐
│   Query Processing      │  ← Analyze intent, detect language
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Embed Query (BGE-M3)  │  ← With instruction prefix
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Vector Search         │  ← Retrieve top 20 candidates
│   (sqlite-vec)          │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Reranker (optional)   │  ← Cross-encoder reranking
│   (BGE-reranker-v2-m3)  │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Diversity Filter      │  ← MMR to reduce redundancy
│   (MMR)                 │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Context Assembly      │  ← Format sources for LLM
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   LLM Generation        │  ← Generate cited response
│   (Claude/GPT/Local)    │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Response Formatting   │  ← Add links, format citations
└─────────────────────────┘
    │
    ▼
Response to User
```

## Functional Requirements

### 1. Query Processing

```python
from dataclasses import dataclass
from typing import Optional, List
from langdetect import detect

@dataclass
class ProcessedQuery:
    original: str
    cleaned: str
    language: str
    query_type: str  # factual, summary, comparison, exploratory
    entities: List[str]
    filters: dict  # Extracted metadata filters

def process_query(user_query: str) -> ProcessedQuery:
    """
    Analyze and enrich the user query.
    """
    # Clean query
    cleaned = user_query.strip()
    
    # Detect language
    language = detect(cleaned) if len(cleaned) > 10 else "en"
    
    # Classify query type
    query_type = classify_query_type(cleaned)
    
    # Extract entities (speakers, topics, videos)
    entities = extract_entities(cleaned)
    
    # Extract implicit filters
    filters = extract_filters(cleaned)
    
    return ProcessedQuery(
        original=user_query,
        cleaned=cleaned,
        language=language,
        query_type=query_type,
        entities=entities,
        filters=filters
    )

def classify_query_type(query: str) -> str:
    """
    Classify query intent for retrieval strategy.
    """
    query_lower = query.lower()
    
    # Summary requests
    if any(kw in query_lower for kw in ["summarize", "summary", "résumé", "résumer", "overview"]):
        return "summary"
    
    # Comparison requests
    if any(kw in query_lower for kw in ["compare", "difference", "vs", "versus", "comparer"]):
        return "comparison"
    
    # Exploratory
    if any(kw in query_lower for kw in ["what are", "list", "tell me about", "quels sont"]):
        return "exploratory"
    
    # Default: factual question
    return "factual"

def extract_filters(query: str) -> dict:
    """
    Extract metadata filters from query.
    """
    filters = {}
    query_lower = query.lower()
    
    # Video-specific queries
    # "in the talk about RAG" → filter by video title
    video_match = re.search(r"(?:in the|dans la?) (?:talk|présentation|video) (?:about|sur) (.+?)(?:\?|$)", query_lower)
    if video_match:
        filters["video_title_contains"] = video_match.group(1)
    
    # Section-specific queries
    if any(kw in query_lower for kw in ["q&a", "question", "demo", "démonstration"]):
        if "q&a" in query_lower or "question" in query_lower:
            filters["section_type"] = "qa"
        elif "demo" in query_lower:
            filters["section_type"] = "demo"
    
    # Language preference
    if "in french" in query_lower or "en français" in query_lower:
        filters["language"] = "fr"
    elif "in english" in query_lower or "en anglais" in query_lower:
        filters["language"] = "en"
    
    return filters
```

**Query type handling strategies:**

| Query Type | Retrieval Strategy | K Value | Notes |
|------------|-------------------|---------|-------|
| `factual` | Standard semantic | 5 | Single focused answer |
| `summary` | Filter by video, get many | 10-15 | Cover full content |
| `comparison` | Multi-query retrieval | 5 per aspect | Compare across sources |
| `exploratory` | Diverse results (high MMR) | 8-10 | Broad coverage |

### 2. Retrieval Configuration

```python
@dataclass
class RetrievalConfig:
    # Initial retrieval
    top_k_initial: int = 20  # Retrieve more for reranking
    
    # After reranking
    top_k_final: int = 5
    
    # Similarity threshold (0-1, higher = stricter)
    similarity_threshold: float = 0.3
    
    # Include surrounding context
    include_context: bool = True
    
    # MMR diversity (0 = no diversity, 1 = max diversity)
    diversity_lambda: float = 0.3
    
    # Hybrid search weight (alpha for dense, 1-alpha for sparse)
    hybrid_alpha: float = 0.7
    
    # Use reranker
    use_reranker: bool = True

# Adjust config based on query type
def get_config_for_query_type(query_type: str) -> RetrievalConfig:
    configs = {
        "factual": RetrievalConfig(top_k_final=5, diversity_lambda=0.2),
        "summary": RetrievalConfig(top_k_final=10, diversity_lambda=0.1),
        "comparison": RetrievalConfig(top_k_final=8, diversity_lambda=0.4),
        "exploratory": RetrievalConfig(top_k_final=8, diversity_lambda=0.5),
    }
    return configs.get(query_type, RetrievalConfig())
```

### 3. Reranking

**Using BGE-reranker-v2-m3:**

```python
from FlagEmbedding import FlagReranker

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.reranker = FlagReranker(
            model_name,
            use_fp16=True,
            device="cuda"
        )
    
    def rerank(
        self, 
        query: str, 
        chunks: List[dict], 
        top_k: int = 5
    ) -> List[dict]:
        """
        Rerank chunks using cross-encoder.
        
        Cross-encoders are more accurate than bi-encoders for ranking
        but slower (can't pre-compute embeddings).
        """
        if not chunks:
            return []
        
        # Prepare pairs for scoring
        pairs = [(query, chunk["text"]) for chunk in chunks]
        
        # Get relevance scores
        scores = self.reranker.compute_score(pairs)
        
        # Sort by score
        ranked = sorted(
            zip(chunks, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top_k with scores
        results = []
        for chunk, score in ranked[:top_k]:
            chunk["rerank_score"] = score
            results.append(chunk)
        
        return results
```

**When to use reranking:**

| Scenario | Use Reranker? | Rationale |
|----------|---------------|-----------|
| High-stakes queries | ✅ Yes | Accuracy matters |
| Simple factual | ⚠️ Optional | Bi-encoder often sufficient |
| Real-time (<1s requirement) | ❌ No | Adds ~200-500ms |
| Batch processing | ✅ Yes | Latency less critical |

### 4. Diversity with MMR

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def maximal_marginal_relevance(
    query_embedding: np.ndarray,
    chunk_embeddings: List[np.ndarray],
    chunks: List[dict],
    lambda_param: float = 0.5,
    top_k: int = 5
) -> List[dict]:
    """
    Maximal Marginal Relevance for diverse results.
    
    Balances relevance to query vs. diversity among selected results.
    
    MMR = λ * sim(doc, query) - (1-λ) * max(sim(doc, selected_docs))
    """
    if len(chunks) <= top_k:
        return chunks
    
    # Compute similarities to query
    query_sims = cosine_similarity(
        [query_embedding], 
        chunk_embeddings
    )[0]
    
    selected_indices = []
    remaining_indices = list(range(len(chunks)))
    
    for _ in range(top_k):
        mmr_scores = []
        
        for idx in remaining_indices:
            # Relevance to query
            relevance = query_sims[idx]
            
            # Max similarity to already selected (diversity penalty)
            if selected_indices:
                selected_embeddings = [chunk_embeddings[i] for i in selected_indices]
                max_sim_to_selected = max(
                    cosine_similarity([chunk_embeddings[idx]], selected_embeddings)[0]
                )
            else:
                max_sim_to_selected = 0
            
            # MMR score
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            mmr_scores.append((idx, mmr))
        
        # Select highest MMR
        best_idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return [chunks[i] for i in selected_indices]
```

### 5. Context Assembly

```python
def build_context(
    chunks: List[dict], 
    include_overlap: bool = True,
    max_context_tokens: int = 4000
) -> str:
    """
    Build context string for LLM prompt.
    """
    context_parts = []
    total_tokens = 0
    
    for i, chunk in enumerate(chunks):
        # Build full text with optional context
        full_text = ""
        
        if include_overlap and chunk.get("context_before"):
            full_text += f"[...] {chunk['context_before']} "
        
        full_text += chunk["text"]
        
        if include_overlap and chunk.get("context_after"):
            full_text += f" {chunk['context_after']} [...]"
        
        # Format timestamp
        start_ts = format_timestamp(chunk["start_time"])
        end_ts = format_timestamp(chunk["end_time"])
        
        # Build source block
        source_block = f"""
═══════════════════════════════════════════
[Source {i + 1}]
Video: {chunk['video_title']}
Timestamp: {start_ts} - {end_ts}
Link: {chunk['youtube_link']}
═══════════════════════════════════════════

{full_text}
"""
        
        # Check token limit
        block_tokens = count_tokens(source_block)
        if total_tokens + block_tokens > max_context_tokens:
            break
        
        context_parts.append(source_block)
        total_tokens += block_tokens
    
    return "\n".join(context_parts)

def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
```

### 6. LLM Generation

**System prompt:**

```python
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Flow Conference talks.

INSTRUCTIONS:
1. Answer based ONLY on the provided sources
2. If the sources don't contain enough information, say so clearly
3. Always cite your sources using [Source N] notation
4. Include video timestamp links when referencing specific content
5. Match the language of the user's question (French or English)
6. Be concise but complete
7. If multiple sources discuss the same topic, synthesize the information

IMPORTANT:
- Never make up information not in the sources
- If you're unsure, say so
- Prefer direct quotes for technical definitions

SOURCES:
{context}"""

USER_PROMPT_TEMPLATE = """Question: {query}

Please answer based on the conference talks provided above. Include relevant source citations and video links."""
```

**Generation with streaming:**

```python
from anthropic import Anthropic

class LLMGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.client = self._init_client()
    
    def _init_client(self):
        provider = self.config.get("provider", "anthropic")
        
        if provider == "anthropic":
            return Anthropic()
        elif provider == "openai":
            from openai import OpenAI
            return OpenAI()
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def generate_streaming(
        self, 
        query: str, 
        context: str
    ) -> AsyncIterator[str]:
        """
        Generate response with streaming.
        """
        system_prompt = SYSTEM_PROMPT.format(context=context)
        user_prompt = USER_PROMPT_TEMPLATE.format(query=query)
        
        if self.config["provider"] == "anthropic":
            async with self.client.messages.stream(
                model=self.config.get("model", "claude-sonnet-4-20250514"),
                max_tokens=self.config.get("max_tokens", 1024),
                temperature=self.config.get("temperature", 0.3),
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            ) as stream:
                async for text in stream.text_stream:
                    yield text
    
    def generate_sync(self, query: str, context: str) -> str:
        """
        Generate response (non-streaming).
        """
        system_prompt = SYSTEM_PROMPT.format(context=context)
        user_prompt = USER_PROMPT_TEMPLATE.format(query=query)
        
        if self.config["provider"] == "anthropic":
            response = self.client.messages.create(
                model=self.config.get("model", "claude-sonnet-4-20250514"),
                max_tokens=self.config.get("max_tokens", 1024),
                temperature=self.config.get("temperature", 0.3),
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
```

### 7. Response Formatting

```python
def format_response(
    response_text: str, 
    chunks: List[dict]
) -> dict:
    """
    Format final response with metadata.
    """
    # Extract cited sources from response
    cited_sources = re.findall(r'\[Source (\d+)\]', response_text)
    cited_indices = set(int(s) - 1 for s in cited_sources)
    
    # Build source list
    sources = []
    for i, chunk in enumerate(chunks):
        if i in cited_indices:
            sources.append({
                "index": i + 1,
                "video_title": chunk["video_title"],
                "timestamp": f"{format_timestamp(chunk['start_time'])} - {format_timestamp(chunk['end_time'])}",
                "youtube_link": chunk["youtube_link"],
                "cited": True
            })
    
    return {
        "answer": response_text,
        "sources": sources,
        "total_sources_retrieved": len(chunks),
        "sources_cited": len(sources)
    }
```

### 8. Fallback Handling

```python
def check_retrieval_quality(
    results: List[dict], 
    threshold: float = 0.3
) -> str:
    """
    Assess retrieval quality for appropriate response.
    """
    if not results:
        return "no_results"
    
    # Check similarity scores
    top_score = results[0].get("similarity", 0)
    
    if top_score < threshold:
        return "low_relevance"
    
    # Check if results are too similar (possible duplication)
    if len(results) > 1:
        scores = [r.get("similarity", 0) for r in results]
        if max(scores) - min(scores) < 0.05:
            return "low_diversity"
    
    return "good"

def generate_fallback_response(
    quality: str, 
    query: str, 
    results: List[dict] = None
) -> str:
    """
    Generate appropriate response for poor retrieval.
    """
    responses = {
        "no_results": f"""I couldn't find any relevant content in the Flow Conference talks for your question about "{query}".

This might be because:
- The topic wasn't covered in the conference
- Try rephrasing your question
- The talks might use different terminology

Would you like to try a different question?""",

        "low_relevance": f"""I found some content that might be related to your question, but I'm not confident it directly addresses what you're asking about.

Here's what I found that seems closest:
{format_low_confidence_results(results)}

Would you like me to search for something more specific?""",

        "low_diversity": """The search results seem to cover very similar content. Let me provide what I found, but note that I might be missing some perspectives on this topic."""
    }
    
    return responses.get(quality, "I encountered an issue processing your question. Please try again.")
```

## Complete Pipeline

```python
class RAGPipeline:
    def __init__(self, config: dict):
        self.embedder = ChunkEmbedder(config["embedding"])
        self.vector_store = VectorStore(config["vector_store"]["db_path"])
        self.reranker = Reranker() if config.get("use_reranker", True) else None
        self.generator = LLMGenerator(config["llm"])
        self.config = config
    
    async def query(self, user_query: str) -> dict:
        """
        Full RAG pipeline.
        """
        # 1. Process query
        processed = process_query(user_query)
        retrieval_config = get_config_for_query_type(processed.query_type)
        
        # 2. Embed query
        query_embedding = self.embedder.embed_query(processed.cleaned)
        
        # 3. Retrieve
        results = self.vector_store.search_with_filters(
            query_embedding=query_embedding["dense"],
            top_k=retrieval_config.top_k_initial,
            **processed.filters
        )
        
        # 4. Check quality
        quality = check_retrieval_quality(results, retrieval_config.similarity_threshold)
        
        if quality in ["no_results", "low_relevance"]:
            return {
                "answer": generate_fallback_response(quality, user_query, results),
                "sources": [],
                "quality": quality
            }
        
        # 5. Rerank
        if self.reranker and retrieval_config.use_reranker:
            results = self.reranker.rerank(
                processed.cleaned, 
                results, 
                top_k=retrieval_config.top_k_final * 2
            )
        
        # 6. Apply MMR for diversity
        if retrieval_config.diversity_lambda > 0:
            embeddings = [self.embedder.embed_query(r["text"])["dense"] for r in results]
            results = maximal_marginal_relevance(
                query_embedding["dense"],
                embeddings,
                results,
                lambda_param=retrieval_config.diversity_lambda,
                top_k=retrieval_config.top_k_final
            )
        else:
            results = results[:retrieval_config.top_k_final]
        
        # 7. Build context
        context = build_context(results, include_overlap=retrieval_config.include_context)
        
        # 8. Generate response
        response_text = self.generator.generate_sync(processed.cleaned, context)
        
        # 9. Format response
        return format_response(response_text, results)
    
    async def query_streaming(self, user_query: str):
        """
        Streaming version for Chainlit.
        """
        # Steps 1-7 same as above...
        
        # Stream generation
        async for token in self.generator.generate_streaming(processed.cleaned, context):
            yield token
```

## Configuration

```yaml
# config/retrieval.yaml
retrieval:
  top_k_initial: 20
  top_k_final: 5
  similarity_threshold: 0.3
  include_context: true
  diversity_lambda: 0.3
  hybrid_alpha: 0.7
  use_reranker: true

reranker:
  model: "BAAI/bge-reranker-v2-m3"
  device: "cuda"
  use_fp16: true

llm:
  provider: "anthropic"  # or "openai", "ollama"
  model: "claude-sonnet-4-20250514"
  temperature: 0.3
  max_tokens: 1024
  streaming: true
```

## Error Handling

```python
class RAGError(Exception):
    """Base exception for RAG pipeline."""
    pass

class EmbeddingError(RAGError):
    """Error during embedding generation."""
    pass

class RetrievalError(RAGError):
    """Error during vector search."""
    pass

class GenerationError(RAGError):
    """Error during LLM generation."""
    pass

async def safe_query(pipeline: RAGPipeline, query: str) -> dict:
    """
    Query with error handling.
    """
    try:
        return await pipeline.query(query)
    except EmbeddingError:
        return {"error": "Failed to process your question. Please try again."}
    except RetrievalError:
        return {"error": "Search service is temporarily unavailable."}
    except GenerationError:
        return {"error": "Failed to generate response. Please try again."}
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return {"error": "An unexpected error occurred."}
```

## Performance Targets

| Stage | Target Latency |
|-------|----------------|
| Query processing | <10ms |
| Embedding | <50ms |
| Vector search | <50ms |
| Reranking (5 docs) | <200ms |
| LLM generation (streaming start) | <500ms |
| **Total (first token)** | **<1s** |
| **Total (complete)** | **<5s** |

## Output

- Generated response string with citations
- Source metadata for UI display
- Quality indicators for monitoring

## Success Criteria

- [ ] Responses are grounded in retrieved sources
- [ ] Citations are accurate and link correctly
- [ ] Fallback handling is graceful
- [ ] Streaming works for responsive UI
- [ ] Latency targets are met
- [ ] Diverse perspectives are included when relevant
