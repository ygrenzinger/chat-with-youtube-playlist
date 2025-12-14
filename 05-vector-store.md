# Component: sqlite-vec Vector Store

## Purpose

Store and retrieve chunk embeddings with metadata using sqlite-vec, providing fast semantic search with zero infrastructure overhead.

## Why sqlite-vec?

| Requirement | sqlite-vec Capability |
|-------------|----------------------|
| No infrastructure | ✅ Single file database |
| Local-first | ✅ No network dependencies |
| Easy backup | ✅ Just copy the .db file |
| Fast for <100k vectors | ✅ Efficient ANN search |
| Metadata filtering | ✅ Standard SQL |
| Python integration | ✅ Built-in sqlite3 + extension |

## Database Schema

```sql
-- Enable sqlite-vec extension (loaded at runtime)
-- .load vec0

-- ============================================
-- VIDEOS TABLE: Metadata about each video
-- ============================================
CREATE TABLE videos (
    video_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    channel TEXT DEFAULT 'Flow Conference',
    upload_date DATE,
    duration_seconds INTEGER,
    language TEXT,
    chunk_count INTEGER DEFAULT 0,
    transcript_quality TEXT CHECK(transcript_quality IN ('manual', 'auto_high', 'auto_low')),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    youtube_url TEXT GENERATED ALWAYS AS ('https://youtube.com/watch?v=' || video_id) STORED
);

-- ============================================
-- CHUNKS TABLE: Text chunks with metadata
-- ============================================
CREATE TABLE chunks (
    chunk_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    section_type TEXT CHECK(section_type IN ('introduction', 'main_content', 'demo', 'qa', 'conclusion')),
    language TEXT,
    token_count INTEGER,
    context_before TEXT,
    context_after TEXT,
    youtube_link TEXT GENERATED ALWAYS AS (
        'https://youtube.com/watch?v=' || video_id || '&t=' || CAST(CAST(start_time AS INTEGER) AS TEXT)
    ) STORED,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);

-- ============================================
-- CHUNK EMBEDDINGS: Dense vectors via sqlite-vec
-- ============================================
CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding FLOAT[1024]  -- BGE-M3 dimension
);

-- ============================================
-- SPARSE EMBEDDINGS: For hybrid search (optional)
-- ============================================
CREATE TABLE chunk_sparse_embeddings (
    chunk_id TEXT PRIMARY KEY,
    indices BLOB NOT NULL,      -- Serialized int array
    values BLOB NOT NULL,       -- Serialized float array
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);

-- ============================================
-- EMBEDDING METADATA: Track versions
-- ============================================
CREATE TABLE embedding_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default metadata
INSERT INTO embedding_metadata (key, value) VALUES 
    ('model_name', 'BAAI/bge-m3'),
    ('model_version', '1.0'),
    ('dimension', '1024'),
    ('schema_version', '1.0');

-- ============================================
-- INDEXES for common queries
-- ============================================
CREATE INDEX idx_chunks_video_id ON chunks(video_id);
CREATE INDEX idx_chunks_section_type ON chunks(section_type);
CREATE INDEX idx_chunks_language ON chunks(language);
CREATE INDEX idx_videos_upload_date ON videos(upload_date DESC);
CREATE INDEX idx_videos_language ON videos(language);

-- ============================================
-- TRIGGERS for updated_at
-- ============================================
CREATE TRIGGER update_chunk_timestamp 
AFTER UPDATE ON chunks
BEGIN
    UPDATE chunks SET updated_at = CURRENT_TIMESTAMP WHERE chunk_id = NEW.chunk_id;
END;
```

## Functional Requirements

### 1. Database Initialization

```python
import sqlite3
from pathlib import Path

class VectorStore:
    def __init__(self, db_path: str = "data/flow_conference.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        
    def connect(self) -> sqlite3.Connection:
        """Connect and load sqlite-vec extension."""
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.enable_load_extension(True)
            
            # Load sqlite-vec extension
            # Path depends on installation method
            try:
                self.conn.load_extension("vec0")
            except Exception:
                # Alternative paths
                import sqlite_vec
                sqlite_vec.load(self.conn)
            
            self.conn.row_factory = sqlite3.Row
        return self.conn
    
    def initialize_schema(self):
        """Create tables if they don't exist."""
        conn = self.connect()
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
```

### 2. Insertion Operations

**Insert video metadata:**

```python
def insert_video(self, video: dict):
    """Insert or update video metadata."""
    conn = self.connect()
    conn.execute("""
        INSERT INTO videos (video_id, title, channel, upload_date, duration_seconds, language, transcript_quality)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            title = excluded.title,
            duration_seconds = excluded.duration_seconds,
            language = excluded.language,
            transcript_quality = excluded.transcript_quality
    """, (
        video["video_id"],
        video["title"],
        video.get("channel", "Flow Conference"),
        video.get("upload_date"),
        video.get("duration_seconds"),
        video.get("language"),
        video.get("transcript_quality")
    ))
    conn.commit()
```

**Insert chunk with embedding:**

```python
import struct
import numpy as np

def serialize_vector(vec: np.ndarray) -> bytes:
    """Serialize numpy array to bytes for sqlite-vec."""
    return struct.pack(f'{len(vec)}f', *vec.astype(np.float32))

def serialize_sparse(indices: list, values: list) -> tuple[bytes, bytes]:
    """Serialize sparse embedding components."""
    indices_bytes = struct.pack(f'{len(indices)}i', *indices)
    values_bytes = struct.pack(f'{len(values)}f', *values)
    return indices_bytes, values_bytes

def insert_chunk(self, chunk: dict, dense_embedding: np.ndarray, sparse_embedding: dict = None):
    """Insert chunk with its embeddings."""
    conn = self.connect()
    
    # Insert chunk metadata
    conn.execute("""
        INSERT INTO chunks (
            chunk_id, video_id, chunk_index, text, start_time, end_time,
            section_type, language, token_count, context_before, context_after
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        chunk["chunk_id"],
        chunk["video_id"],
        chunk["chunk_index"],
        chunk["text"],
        chunk["start_time"],
        chunk["end_time"],
        chunk.get("section_type"),
        chunk.get("language"),
        chunk.get("token_count"),
        chunk.get("context_before"),
        chunk.get("context_after")
    ))
    
    # Insert dense embedding
    conn.execute("""
        INSERT INTO chunk_embeddings (chunk_id, embedding)
        VALUES (?, ?)
    """, (chunk["chunk_id"], serialize_vector(dense_embedding)))
    
    # Insert sparse embedding if provided
    if sparse_embedding:
        indices_bytes, values_bytes = serialize_sparse(
            sparse_embedding["indices"],
            sparse_embedding["values"]
        )
        conn.execute("""
            INSERT INTO chunk_sparse_embeddings (chunk_id, indices, values)
            VALUES (?, ?, ?)
        """, (chunk["chunk_id"], indices_bytes, values_bytes))
    
    conn.commit()

def insert_chunks_batch(self, chunks: list, embeddings: list, batch_size: int = 100):
    """Batch insert for efficiency."""
    conn = self.connect()
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        
        with conn:  # Transaction
            for chunk, emb in zip(batch_chunks, batch_embeddings):
                self.insert_chunk(chunk, emb["dense"], emb.get("sparse"))
    
    # Update chunk counts
    conn.execute("""
        UPDATE videos SET chunk_count = (
            SELECT COUNT(*) FROM chunks WHERE chunks.video_id = videos.video_id
        )
    """)
    conn.commit()
```

### 3. Vector Search

**Basic semantic search:**

```python
def search_similar(
    self, 
    query_embedding: np.ndarray, 
    top_k: int = 10,
    threshold: float = None
) -> list:
    """
    Find similar chunks using cosine distance.
    
    Returns list of dicts with chunk data and distance score.
    """
    conn = self.connect()
    
    query_vec = serialize_vector(query_embedding)
    
    sql = """
        SELECT 
            c.chunk_id,
            c.text,
            c.video_id,
            c.start_time,
            c.end_time,
            c.section_type,
            c.context_before,
            c.context_after,
            c.youtube_link,
            v.title as video_title,
            vec_distance_cosine(e.embedding, ?) as distance
        FROM chunk_embeddings e
        JOIN chunks c ON e.chunk_id = c.chunk_id
        JOIN videos v ON c.video_id = v.video_id
        ORDER BY distance ASC
        LIMIT ?
    """
    
    cursor = conn.execute(sql, (query_vec, top_k))
    results = [dict(row) for row in cursor.fetchall()]
    
    # Apply threshold filter if specified
    if threshold is not None:
        results = [r for r in results if r["distance"] <= threshold]
    
    # Convert distance to similarity score (1 - distance for cosine)
    for r in results:
        r["similarity"] = 1 - r["distance"]
    
    return results
```

**Filtered search:**

```python
def search_with_filters(
    self,
    query_embedding: np.ndarray,
    top_k: int = 10,
    video_id: str = None,
    section_type: str = None,
    language: str = None,
    min_date: str = None,
    max_date: str = None
) -> list:
    """
    Search with metadata filters.
    """
    conn = self.connect()
    
    # Build WHERE clause dynamically
    conditions = []
    params = [serialize_vector(query_embedding)]
    
    if video_id:
        conditions.append("c.video_id = ?")
        params.append(video_id)
    
    if section_type:
        conditions.append("c.section_type = ?")
        params.append(section_type)
    
    if language:
        conditions.append("c.language = ?")
        params.append(language)
    
    if min_date:
        conditions.append("v.upload_date >= ?")
        params.append(min_date)
    
    if max_date:
        conditions.append("v.upload_date <= ?")
        params.append(max_date)
    
    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)
    
    params.append(top_k)
    
    sql = f"""
        SELECT 
            c.chunk_id,
            c.text,
            c.video_id,
            c.start_time,
            c.end_time,
            c.youtube_link,
            v.title as video_title,
            vec_distance_cosine(e.embedding, ?) as distance
        FROM chunk_embeddings e
        JOIN chunks c ON e.chunk_id = c.chunk_id
        JOIN videos v ON c.video_id = v.video_id
        {where_clause}
        ORDER BY distance ASC
        LIMIT ?
    """
    
    cursor = conn.execute(sql, params)
    return [dict(row) for row in cursor.fetchall()]
```

### 4. Hybrid Search (Dense + Sparse)

```python
def hybrid_search(
    self,
    query_dense: np.ndarray,
    query_sparse: dict,
    top_k: int = 10,
    alpha: float = 0.7  # Weight for dense (1-alpha for sparse)
) -> list:
    """
    Combine dense and sparse search results.
    
    Uses Reciprocal Rank Fusion (RRF) for score combination.
    """
    # Get dense results (fetch more for fusion)
    dense_results = self.search_similar(query_dense, top_k=top_k * 3)
    
    # Get sparse results
    sparse_results = self._sparse_search(query_sparse, top_k=top_k * 3)
    
    # Reciprocal Rank Fusion
    k = 60  # RRF constant
    scores = {}
    
    for rank, result in enumerate(dense_results):
        chunk_id = result["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + alpha * (1 / (k + rank + 1))
        # Store result data for later
        if chunk_id not in scores:
            scores[chunk_id] = {"data": result, "score": 0}
    
    for rank, result in enumerate(sparse_results):
        chunk_id = result["chunk_id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + (1 - alpha) * (1 / (k + rank + 1))
    
    # Sort by combined score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Fetch full data for top results
    top_ids = [chunk_id for chunk_id, _ in ranked[:top_k]]
    return self._fetch_chunks_by_ids(top_ids)

def _sparse_search(self, query_sparse: dict, top_k: int) -> list:
    """
    BM25-style sparse search.
    
    Note: This is a simplified implementation.
    For production, consider using FTS5 or dedicated sparse index.
    """
    conn = self.connect()
    
    # This requires custom scoring logic
    # Simplified: use SQL LIKE for key terms (not ideal)
    # Better: implement proper BM25 or use FTS5
    ...
```

### 5. CRUD Operations

```python
def get_chunk(self, chunk_id: str) -> dict:
    """Get single chunk by ID."""
    conn = self.connect()
    cursor = conn.execute("""
        SELECT c.*, v.title as video_title
        FROM chunks c
        JOIN videos v ON c.video_id = v.video_id
        WHERE c.chunk_id = ?
    """, (chunk_id,))
    row = cursor.fetchone()
    return dict(row) if row else None

def get_video_chunks(self, video_id: str) -> list:
    """Get all chunks for a video, ordered."""
    conn = self.connect()
    cursor = conn.execute("""
        SELECT * FROM chunks
        WHERE video_id = ?
        ORDER BY chunk_index
    """, (video_id,))
    return [dict(row) for row in cursor.fetchall()]

def delete_video(self, video_id: str):
    """Delete video and all associated data."""
    conn = self.connect()
    
    # Get chunk IDs first
    cursor = conn.execute(
        "SELECT chunk_id FROM chunks WHERE video_id = ?", 
        (video_id,)
    )
    chunk_ids = [row[0] for row in cursor.fetchall()]
    
    # Delete embeddings
    for chunk_id in chunk_ids:
        conn.execute(
            "DELETE FROM chunk_embeddings WHERE chunk_id = ?",
            (chunk_id,)
        )
        conn.execute(
            "DELETE FROM chunk_sparse_embeddings WHERE chunk_id = ?",
            (chunk_id,)
        )
    
    # Delete chunks (cascade would handle this if FK enforced)
    conn.execute("DELETE FROM chunks WHERE video_id = ?", (video_id,))
    
    # Delete video
    conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
    
    conn.commit()

def update_chunk_text(self, chunk_id: str, new_text: str, new_embedding: np.ndarray):
    """Update chunk text and re-embed."""
    conn = self.connect()
    
    conn.execute(
        "UPDATE chunks SET text = ?, updated_at = CURRENT_TIMESTAMP WHERE chunk_id = ?",
        (new_text, chunk_id)
    )
    
    conn.execute(
        "UPDATE chunk_embeddings SET embedding = ? WHERE chunk_id = ?",
        (serialize_vector(new_embedding), chunk_id)
    )
    
    conn.commit()
```

### 6. Statistics and Monitoring

```python
def get_stats(self) -> dict:
    """Get database statistics."""
    conn = self.connect()
    
    stats = {}
    
    # Total counts
    stats["total_videos"] = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    stats["total_chunks"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    
    # By language
    cursor = conn.execute("""
        SELECT language, COUNT(*) as count 
        FROM chunks 
        GROUP BY language
    """)
    stats["chunks_by_language"] = dict(cursor.fetchall())
    
    # By section type
    cursor = conn.execute("""
        SELECT section_type, COUNT(*) as count 
        FROM chunks 
        GROUP BY section_type
    """)
    stats["chunks_by_section"] = dict(cursor.fetchall())
    
    # Average chunk size
    stats["avg_token_count"] = conn.execute(
        "SELECT AVG(token_count) FROM chunks"
    ).fetchone()[0]
    
    # Database file size
    stats["db_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)
    
    return stats
```

## Performance Optimization

### Indexing Strategy

For <100k vectors, sqlite-vec's default exact search is fast enough (~50ms).

For larger collections:
```sql
-- Create IVF index for approximate search
-- Note: Syntax may vary by sqlite-vec version
CREATE INDEX chunk_embeddings_ivf ON chunk_embeddings 
USING ivf(embedding, 100);  -- 100 clusters
```

### Query Optimization

```python
# Use LIMIT to avoid fetching too many results
# Always specify columns instead of SELECT *
# Use indexes for filtered queries

# Good: Specific columns, filtered, limited
SELECT c.chunk_id, c.text, c.youtube_link
FROM chunks c
WHERE c.language = 'fr'
LIMIT 10

# Bad: Full table scan
SELECT * FROM chunks
```

### Connection Pooling

```python
from contextlib import contextmanager
import threading

class ConnectionPool:
    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool = []
        self.lock = threading.Lock()
        
    @contextmanager
    def get_connection(self):
        with self.lock:
            if self.pool:
                conn = self.pool.pop()
            else:
                conn = self._create_connection()
        try:
            yield conn
        finally:
            with self.lock:
                self.pool.append(conn)
```

## Performance Targets

| Operation | Target Latency | Notes |
|-----------|----------------|-------|
| Vector search (10k chunks) | <50ms | Exact search |
| Vector search (100k chunks) | <200ms | May need ANN index |
| Batch insert (100 chunks) | <500ms | With transaction |
| Metadata lookup | <10ms | Indexed |
| Full-text filter + vector | <100ms | Combined query |

## Backup and Maintenance

```python
def backup_database(self, backup_path: str):
    """Create backup of database."""
    import shutil
    self.close()  # Ensure no writes in progress
    shutil.copy2(self.db_path, backup_path)

def vacuum_database(self):
    """Reclaim space and optimize."""
    conn = self.connect()
    conn.execute("VACUUM")

def integrity_check(self) -> bool:
    """Check database integrity."""
    conn = self.connect()
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    return result == "ok"
```

## Configuration

```yaml
# config/vector_store.yaml
vector_store:
  db_path: "data/flow_conference.db"
  
  search:
    default_top_k: 10
    max_top_k: 100
    default_threshold: null
    
  hybrid:
    enabled: true
    dense_weight: 0.7  # alpha
    
  performance:
    use_wal_mode: true
    cache_size_mb: 64
    
  backup:
    enabled: true
    frequency: "daily"
    keep_count: 7
```

## Dependencies

```python
# requirements.txt
sqlite-vec>=0.1.0
numpy>=1.24.0
```

## Output Location

- Database file: `data/flow_conference.db`
- Backups: `data/backups/flow_conference_{timestamp}.db`

## Success Criteria

- [ ] Schema creates correctly with all tables
- [ ] Vector search returns relevant results
- [ ] Filtered search works with metadata
- [ ] Batch inserts are performant
- [ ] Database integrity is maintained
- [ ] Backup/restore works correctly
