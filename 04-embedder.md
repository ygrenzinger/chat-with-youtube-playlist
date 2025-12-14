# Component: BGE-M3 Embedder

## Purpose

Generate embeddings for all chunks using BGE-M3, producing dense vectors suitable for semantic search, with optional sparse vectors for hybrid retrieval.

## Input

Chunked transcript data with text and metadata.

## Output

Embeddings added to chunk data:

```json
{
  "chunk_id": "abc123xyz_001",
  "text": "...",
  "dense_embedding": [0.023, -0.156, ...],
  "sparse_embedding": {
    "indices": [102, 3847, 9821, ...],
    "values": [0.45, 0.32, 0.28, ...]
  },
  "embedding_model": "BAAI/bge-m3",
  "embedding_version": "1.0"
}
```

## BGE-M3 Model Overview

### Specifications

| Property | Value |
|----------|-------|
| Model Name | `BAAI/bge-m3` |
| Dense Dimension | 1024 |
| Max Sequence Length | 8192 tokens |
| Languages Supported | 100+ (including French and English) |
| Embedding Types | Dense, Sparse, ColBERT (multi-vector) |
| Model Size | ~2.3GB |

### Why BGE-M3 for This Project?

| Requirement | BGE-M3 Capability |
|-------------|-------------------|
| French + English content | ✅ Multilingual training |
| Long chunks (350 tokens) | ✅ 8192 token context |
| Hybrid search | ✅ Dense + Sparse outputs |
| Local deployment | ✅ Open weights |
| Quality | ✅ State-of-the-art on MTEB |

## Functional Requirements

### 1. Model Configuration

```python
from FlagEmbedding import BGEM3FlagModel

model_config = {
    "model_name": "BAAI/bge-m3",
    "device": "cuda",  # or "cpu", "mps"
    "normalize_embeddings": True,
    "use_fp16": True,  # Half precision for faster inference
    "batch_size": 32,  # Adjust based on GPU memory
}

# Batch size recommendations by GPU memory
BATCH_SIZES = {
    "8GB": 8,
    "16GB": 32,
    "24GB": 64,
    "CPU": 4,
}
```

### 2. Embedding Generation

**For chunks (indexing):**

```python
def embed_chunks(chunks: List[Chunk], model: BGEM3FlagModel) -> List[dict]:
    """
    Embed chunks for indexing.
    Only embed the main text field, not context_before/after.
    """
    texts = [chunk.text for chunk in chunks]
    
    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=32,
        max_length=512,  # Sufficient for ~350 token chunks
        return_dense=True,
        return_sparse=True,  # For hybrid search
        return_colbert_vecs=False,  # Not needed for basic retrieval
    )
    
    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "chunk_id": chunk.chunk_id,
            "dense_embedding": embeddings["dense_vecs"][i].tolist(),
            "sparse_embedding": {
                "indices": embeddings["lexical_weights"][i].indices.tolist(),
                "values": embeddings["lexical_weights"][i].values.tolist(),
            }
        })
    
    return results
```

**For queries (runtime):**

```python
def embed_query(query: str, model: BGEM3FlagModel) -> dict:
    """
    Embed user query for retrieval.
    BGE-M3 recommends instruction prefix for queries.
    """
    # Instruction prefix improves retrieval quality
    instruction = "Represent this sentence for searching relevant passages: "
    
    embedding = model.encode(
        [instruction + query],
        max_length=256,  # Queries are typically shorter
        return_dense=True,
        return_sparse=True,
    )
    
    return {
        "dense": embedding["dense_vecs"][0],
        "sparse": {
            "indices": embedding["lexical_weights"][0].indices.tolist(),
            "values": embedding["lexical_weights"][0].values.tolist(),
        }
    }
```

### 3. Embedding Types Explained

**Dense Embeddings (Primary):**
- 1024-dimensional float vectors
- Capture semantic meaning
- Good for conceptual similarity
- Example: "RAG system" ≈ "retrieval augmented generation"

**Sparse Embeddings (Hybrid):**
- Variable-length token-weight pairs
- Capture lexical/keyword matching
- Good for exact term matching
- Example: "BGE-M3" matches exactly

**When to use each:**

| Query Type | Best Approach |
|------------|---------------|
| Conceptual questions | Dense only |
| Technical terms, names | Dense + Sparse hybrid |
| Exact quotes | Sparse-weighted |
| General questions | Dense with sparse boost |

### 4. Batch Processing

```python
class ChunkEmbedder:
    def __init__(self, config: dict):
        self.model = None
        self.config = config
        
    def _load_model(self):
        """Lazy load model to avoid memory issues."""
        if self.model is None:
            from FlagEmbedding import BGEM3FlagModel
            self.model = BGEM3FlagModel(
                self.config["model_name"],
                use_fp16=self.config.get("use_fp16", True),
                device=self.config.get("device", "cuda")
            )
        return self.model
    
    def embed_all_chunks(
        self, 
        chunks: List[Chunk], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[dict]:
        """
        Embed all chunks with batching and progress tracking.
        """
        model = self._load_model()
        results = []
        
        # Process in batches
        batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
        
        for batch in tqdm(batches, disable=not show_progress):
            batch_embeddings = self.embed_chunks(batch, model)
            results.extend(batch_embeddings)
        
        return results
    
    def embed_incrementally(
        self,
        chunks: List[Chunk],
        existing_ids: set,
        batch_size: int = 32
    ) -> List[dict]:
        """
        Only embed chunks not already in the database.
        """
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        
        if not new_chunks:
            return []
        
        return self.embed_all_chunks(new_chunks, batch_size)
```

### 5. Storage Format

**Dense embeddings:**

| Format | Use Case | Size per Vector |
|--------|----------|-----------------|
| NumPy `.npy` | Intermediate processing | 4KB (float32) |
| JSON list | Portability | ~8KB (string) |
| Binary blob | sqlite-vec storage | 4KB |

**Sparse embeddings:**

```json
{
  "indices": [102, 3847, 9821, 15234],
  "values": [0.45, 0.32, 0.28, 0.15]
}
```

### 6. Incremental Updates

```python
class EmbeddingManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.version = "1.0"
    
    def get_embedded_chunk_ids(self) -> set:
        """Get IDs of already-embedded chunks."""
        # Query database for existing chunk_ids
        ...
    
    def should_reembed(self, chunk: Chunk, stored_version: str) -> bool:
        """Check if chunk needs re-embedding."""
        # Re-embed if:
        # - Model version changed
        # - Chunk text changed
        return stored_version != self.version
    
    def mark_for_reembedding(self, chunk_ids: List[str]):
        """Mark chunks for re-embedding (e.g., after model update)."""
        ...
```

## Performance Optimization

### GPU Memory Management

```python
import torch

def get_optimal_batch_size() -> int:
    """Determine batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 4  # CPU fallback
    
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    if gpu_memory_gb >= 24:
        return 64
    elif gpu_memory_gb >= 16:
        return 32
    elif gpu_memory_gb >= 8:
        return 8
    else:
        return 4

def clear_gpu_memory():
    """Clear GPU cache between large batches."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Parallel Processing (CPU)

```python
from concurrent.futures import ThreadPoolExecutor

def embed_parallel_cpu(chunks: List[Chunk], num_workers: int = 4) -> List[dict]:
    """Parallel embedding on CPU with multiple workers."""
    # Note: Model loading happens per-worker, memory intensive
    # Only use if you have sufficient RAM
    ...
```

## Performance Targets

| Metric | GPU (RTX 3090) | GPU (RTX 3060) | CPU |
|--------|----------------|----------------|-----|
| Chunks/second | 100+ | 50+ | 5+ |
| Memory per chunk | ~4KB | ~4KB | ~4KB |
| Model load time | ~10s | ~10s | ~30s |
| Total for 1000 chunks | ~4MB | ~4MB | ~4MB |

## Configuration

```yaml
# config/embedding.yaml
embedding:
  model_name: "BAAI/bge-m3"
  device: "auto"  # auto-detect GPU/CPU
  use_fp16: true
  normalize: true
  
  dense:
    enabled: true
    dimension: 1024
    
  sparse:
    enabled: true  # For hybrid search
    
  colbert:
    enabled: false  # Not needed for basic RAG
    
  batch_processing:
    batch_size: "auto"  # or specific number
    show_progress: true
    checkpoint_every: 100  # Save progress
    
  versioning:
    current_version: "1.0"
    track_changes: true
```

## Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Very short chunks (<50 tokens) | Embed as-is, may have lower quality |
| Non-text in chunk (leaked timestamps) | Clean before embedding |
| Mixed language chunks | BGE-M3 handles natively |
| Special characters/emoji | Tokenizer handles most cases |
| GPU out of memory | Reduce batch size, use CPU fallback |

## Dependencies

```python
# requirements.txt
FlagEmbedding>=1.2.0
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
```

## Example Usage

```python
from embedder import ChunkEmbedder
from config import load_config

# Load configuration
config = load_config("config/embedding.yaml")

# Initialize embedder
embedder = ChunkEmbedder(config)

# Load chunks from previous step
chunks = load_chunks("data/chunks/")

# Embed all chunks
embeddings = embedder.embed_all_chunks(chunks)

# Save embeddings (will be loaded into sqlite-vec)
save_embeddings(embeddings, "data/embeddings/")
```

## Output Location

- Embeddings stored directly in sqlite-vec database
- Backup: `data/embeddings/{video_id}_embeddings.npy`

## Success Criteria

- [ ] All chunks are embedded with consistent dimensions
- [ ] Embedding version is tracked for updates
- [ ] Batch processing doesn't run out of memory
- [ ] Incremental updates work correctly
- [ ] Both dense and sparse embeddings are generated
- [ ] Query embedding includes instruction prefix
