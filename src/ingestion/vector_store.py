"""
Vector Store using SQLite with sqlite-vector extension.

Stores chunks and embeddings for efficient similarity search in RAG pipelines.
"""

import importlib.resources
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.ingestion.chunker import (
    Chunk,
    ChunkedVideo,
    GeminiEmbedding,
    ParentChunk,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for the vector store."""
    db_path: Path = Path("data/vector_store.db")
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    quantize_after_insert: bool = True


@dataclass
class SearchResult:
    """Result from a vector similarity search."""
    chunk_id: str
    video_id: str
    video_title: str
    text: str
    youtube_link: str
    start_time: float
    end_time: float
    distance: float
    parent_chunk_id: str = ""


@dataclass
class SearchResultWithContext:
    """Search result with parent chunk for expanded LLM context."""
    chunk: SearchResult
    parent_text: str
    parent_chunk_id: str


class VectorStore:
    """
    SQLite-based vector store using sqlite-vector extension.

    Stores child chunks with embeddings for similarity search,
    and parent chunks for context expansion.
    """

    SCHEMA = """
    -- Videos table
    CREATE TABLE IF NOT EXISTS videos (
        video_id TEXT PRIMARY KEY,
        video_title TEXT NOT NULL,
        total_chunks INTEGER DEFAULT 0,
        total_parent_chunks INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- Parent chunks (NOT embedded - used for LLM context expansion)
    CREATE TABLE IF NOT EXISTS parent_chunks (
        parent_chunk_id TEXT PRIMARY KEY,
        parent_index INTEGER NOT NULL,
        video_id TEXT NOT NULL,
        text TEXT NOT NULL,
        start_time REAL NOT NULL,
        end_time REAL NOT NULL,
        token_count INTEGER NOT NULL,
        FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
    );

    -- Child chunks (embedded - used for vector search)
    CREATE TABLE IF NOT EXISTS child_chunks (
        chunk_id TEXT PRIMARY KEY,
        chunk_index INTEGER NOT NULL,
        video_id TEXT NOT NULL,
        parent_chunk_id TEXT,
        text TEXT NOT NULL,
        start_time REAL NOT NULL,
        end_time REAL NOT NULL,
        token_count INTEGER NOT NULL,
        youtube_link TEXT NOT NULL,
        context_before TEXT DEFAULT '',
        context_after TEXT DEFAULT '',
        embedding BLOB,
        FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
        FOREIGN KEY (parent_chunk_id) REFERENCES parent_chunks(parent_chunk_id) ON DELETE SET NULL
    );

    -- Indexes for faster lookups
    CREATE INDEX IF NOT EXISTS idx_child_chunks_video ON child_chunks(video_id);
    CREATE INDEX IF NOT EXISTS idx_parent_chunks_video ON parent_chunks(video_id);
    CREATE INDEX IF NOT EXISTS idx_child_chunks_parent ON child_chunks(parent_chunk_id);
    """

    def __init__(self, config: VectorStoreConfig | None = None):
        """
        Initialize the vector store.

        Args:
            config: Vector store configuration
        """
        self.config = config or VectorStoreConfig()
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection | None = None
        self._embedding_client: GeminiEmbedding | None = None
        self._vector_initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.config.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._load_extension()
            self._init_schema()
            self._init_vector_index()
        return self._conn

    def _load_extension(self) -> None:
        """Load the sqlite-vector extension."""
        try:
            ext_path = importlib.resources.files("sqlite_vector.binaries") / "vector"
            self._conn.enable_load_extension(True)
            self._conn.load_extension(str(ext_path))
            self._conn.enable_load_extension(False)

            version = self._conn.execute("SELECT vector_version()").fetchone()[0]
            logger.info(f"Loaded sqlite-vector extension v{version}")
        except Exception as e:
            raise RuntimeError(f"Failed to load sqlite-vector extension: {e}")

    def _init_schema(self) -> None:
        """Initialize the database schema."""
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()
        logger.debug("Database schema initialized")

    def _init_vector_index(self) -> None:
        """Initialize the vector index on embeddings column."""
        if self._vector_initialized:
            return

        conn = self._get_connection()
        try:
            conn.execute(f"""
                SELECT vector_init('child_chunks', 'embedding',
                    'dimension={self.config.embedding_dimensions},type=FLOAT32,distance=COSINE')
            """)
            conn.commit()
            self._vector_initialized = True
            logger.info("Vector index initialized")
        except sqlite3.OperationalError as e:
            if "already initialized" in str(e).lower():
                self._vector_initialized = True
            else:
                raise

    def _get_embedding_client(self) -> GeminiEmbedding:
        """Get or create the embedding client."""
        if self._embedding_client is None:
            self._embedding_client = GeminiEmbedding(
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dimensions,
                task_type="RETRIEVAL_DOCUMENT",
            )
        return self._embedding_client

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # -------------------------------------------------------------------------
    # Insert Operations
    # -------------------------------------------------------------------------

    def insert_video(
        self,
        video: ChunkedVideo,
        embeddings: np.ndarray | None = None,
        compute_embeddings: bool = True,
    ) -> None:
        """
        Insert a video with all its chunks into the store.

        Args:
            video: ChunkedVideo object with chunks and parent_chunks
            embeddings: Pre-computed embeddings (optional)
            compute_embeddings: Whether to compute embeddings if not provided
        """
        conn = self._get_connection()

        # Check if video already exists
        existing = conn.execute(
            "SELECT video_id FROM videos WHERE video_id = ?",
            (video.video_id,)
        ).fetchone()

        if existing:
            logger.warning(f"Video {video.video_id} already exists, skipping")
            return

        # Compute embeddings if needed
        if embeddings is None and compute_embeddings and video.chunks:
            logger.info(f"Computing embeddings for {len(video.chunks)} chunks...")
            texts = [chunk.text for chunk in video.chunks]
            embeddings = self._get_embedding_client().embed(texts)

        # Insert video metadata
        conn.execute("""
            INSERT INTO videos (video_id, video_title, total_chunks, total_parent_chunks)
            VALUES (?, ?, ?, ?)
        """, (video.video_id, video.video_title, video.total_chunks, video.total_parent_chunks))

        # Insert parent chunks first (for foreign key references)
        for parent in video.parent_chunks:
            self._insert_parent_chunk(conn, parent)

        # Insert child chunks with embeddings
        for i, chunk in enumerate(video.chunks):
            embedding = embeddings[i] if embeddings is not None else None
            self._insert_chunk(conn, chunk, embedding)

        conn.commit()

        # Initialize vector index after first insert
        self._init_vector_index()

        logger.info(
            f"Inserted video {video.video_id}: "
            f"{video.total_chunks} chunks, {video.total_parent_chunks} parent chunks"
        )

    def _insert_chunk(
        self,
        conn: sqlite3.Connection,
        chunk: Chunk,
        embedding: np.ndarray | None,
    ) -> None:
        """Insert a single chunk with its embedding."""
        blob = embedding.astype(np.float32).tobytes() if embedding is not None else None

        conn.execute("""
            INSERT INTO child_chunks
            (chunk_id, chunk_index, video_id, parent_chunk_id, text,
             start_time, end_time, token_count, youtube_link,
             context_before, context_after, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk.chunk_id,
            chunk.chunk_index,
            chunk.video_id,
            chunk.parent_chunk_id or None,
            chunk.text,
            chunk.start_time,
            chunk.end_time,
            chunk.token_count,
            chunk.youtube_link,
            chunk.context_before,
            chunk.context_after,
            blob,
        ))

    def _insert_parent_chunk(
        self,
        conn: sqlite3.Connection,
        parent: ParentChunk,
    ) -> None:
        """Insert a single parent chunk."""
        conn.execute("""
            INSERT INTO parent_chunks
            (parent_chunk_id, parent_index, video_id, text,
             start_time, end_time, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            parent.parent_chunk_id,
            parent.parent_index,
            parent.video_id,
            parent.text,
            parent.start_time,
            parent.end_time,
            parent.token_count,
        ))

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    def search(
        self,
        query: str | np.ndarray,
        k: int = 10,
        use_quantized: bool = True,
    ) -> list[SearchResult]:
        """
        Search for similar chunks.

        Args:
            query: Query text or pre-computed embedding
            k: Number of results to return
            use_quantized: Use quantized search (faster, approximate)

        Returns:
            List of SearchResult objects sorted by distance
        """
        conn = self._get_connection()

        # Get query embedding
        if isinstance(query, str):
            query_client = GeminiEmbedding(
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dimensions,
                task_type="RETRIEVAL_QUERY",
            )
            query_embedding = query_client.embed([query])[0]
        else:
            query_embedding = query

        query_blob = query_embedding.astype(np.float32).tobytes()

        # Choose search function
        search_fn = "vector_quantize_scan" if use_quantized else "vector_full_scan"

        try:
            rows = conn.execute(f"""
                SELECT c.chunk_id, c.video_id, c.text, c.youtube_link,
                       c.start_time, c.end_time, c.parent_chunk_id,
                       v.video_title, s.distance
                FROM child_chunks c
                JOIN videos v ON c.video_id = v.video_id
                JOIN {search_fn}('child_chunks', 'embedding', ?, ?) s
                    ON c.rowid = s.rowid
                ORDER BY s.distance ASC
            """, (query_blob, k)).fetchall()
        except sqlite3.OperationalError as e:
            if "quantize" in str(e).lower() and use_quantized:
                logger.warning("Quantization not available, falling back to full scan")
                return self.search(query, k, use_quantized=False)
            raise

        return [
            SearchResult(
                chunk_id=row["chunk_id"],
                video_id=row["video_id"],
                video_title=row["video_title"],
                text=row["text"],
                youtube_link=row["youtube_link"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                distance=row["distance"],
                parent_chunk_id=row["parent_chunk_id"] or "",
            )
            for row in rows
        ]

    def search_with_context(
        self,
        query: str | np.ndarray,
        k: int = 10,
    ) -> list[SearchResultWithContext]:
        """
        Search with parent chunk context expansion.

        Args:
            query: Query text or pre-computed embedding
            k: Number of results to return

        Returns:
            List of SearchResultWithContext with parent text for LLM
        """
        results = self.search(query, k)
        conn = self._get_connection()

        results_with_context = []
        for result in results:
            parent_text = result.text  # Default to chunk text

            if result.parent_chunk_id:
                parent_row = conn.execute(
                    "SELECT text FROM parent_chunks WHERE parent_chunk_id = ?",
                    (result.parent_chunk_id,)
                ).fetchone()
                if parent_row:
                    parent_text = parent_row["text"]

            results_with_context.append(SearchResultWithContext(
                chunk=result,
                parent_text=parent_text,
                parent_chunk_id=result.parent_chunk_id,
            ))

        return results_with_context

    # -------------------------------------------------------------------------
    # Utility Operations
    # -------------------------------------------------------------------------

    def quantize(self) -> int:
        """
        Run quantization for faster approximate search.

        Returns:
            Number of rows quantized
        """
        conn = self._get_connection()
        result = conn.execute(
            "SELECT vector_quantize('child_chunks', 'embedding')"
        ).fetchone()[0]
        conn.commit()
        logger.info(f"Quantized {result} rows")
        return result

    def get_stats(self) -> dict:
        """Get database statistics."""
        conn = self._get_connection()

        stats = {
            "videos": conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0],
            "child_chunks": conn.execute("SELECT COUNT(*) FROM child_chunks").fetchone()[0],
            "parent_chunks": conn.execute("SELECT COUNT(*) FROM parent_chunks").fetchone()[0],
            "chunks_with_embeddings": conn.execute(
                "SELECT COUNT(*) FROM child_chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0],
        }

        # Get vector extension info
        try:
            stats["vector_version"] = conn.execute("SELECT vector_version()").fetchone()[0]
            stats["vector_backend"] = conn.execute("SELECT vector_backend()").fetchone()[0]
        except Exception:
            pass

        return stats

    def delete_video(self, video_id: str) -> bool:
        """
        Delete a video and all its chunks.

        Args:
            video_id: Video ID to delete

        Returns:
            True if video was deleted, False if not found
        """
        conn = self._get_connection()

        # Check if exists
        existing = conn.execute(
            "SELECT video_id FROM videos WHERE video_id = ?",
            (video_id,)
        ).fetchone()

        if not existing:
            return False

        # Delete cascades to chunks due to foreign keys
        conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        conn.commit()

        logger.info(f"Deleted video {video_id}")
        return True

    def list_videos(self) -> list[dict]:
        """List all videos in the store."""
        conn = self._get_connection()
        rows = conn.execute("""
            SELECT video_id, video_title, total_chunks, total_parent_chunks, created_at
            FROM videos ORDER BY created_at DESC
        """).fetchall()

        return [dict(row) for row in rows]


# -----------------------------------------------------------------------------
# Ingestion Functions
# -----------------------------------------------------------------------------

def load_chunked_video(json_path: Path) -> ChunkedVideo:
    """Load a ChunkedVideo from a JSON file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))

    chunks = [
        Chunk(
            chunk_id=c["chunk_id"],
            chunk_index=c["chunk_index"],
            video_id=c["video_id"],
            video_title=c["video_title"],
            text=c["text"],
            start_time=c["start_time"],
            end_time=c["end_time"],
            token_count=c["token_count"],
            section_type=c.get("section_type", "main_content"),
            context_before=c.get("context_before", ""),
            context_after=c.get("context_after", ""),
            youtube_link=c.get("youtube_link", ""),
            parent_chunk_id=c.get("parent_chunk_id", ""),
            youtube_video_id=c.get("youtube_video_id", ""),
        )
        for c in data.get("chunks", [])
    ]

    parent_chunks = [
        ParentChunk(
            parent_chunk_id=p["parent_chunk_id"],
            parent_index=p["parent_index"],
            video_id=p["video_id"],
            video_title=p["video_title"],
            text=p["text"],
            start_time=p["start_time"],
            end_time=p["end_time"],
            token_count=p["token_count"],
            child_chunk_ids=p.get("child_chunk_ids", []),
        )
        for p in data.get("parent_chunks", [])
    ]

    return ChunkedVideo(
        video_id=data["video_id"],
        video_title=data["video_title"],
        total_chunks=data["total_chunks"],
        chunks=chunks,
        total_parent_chunks=data.get("total_parent_chunks", 0),
        parent_chunks=parent_chunks,
    )


def ingest_chunks(
    input_path: Path | None = None,
    input_dir: Path | None = None,
    db_path: Path = Path("data/vector_store.db"),
) -> None:
    """
    Ingest chunk files into the vector store.

    Args:
        input_path: Single chunk JSON file
        input_dir: Directory containing chunk JSON files
        db_path: Path to the vector store database
    """
    config = VectorStoreConfig(db_path=db_path)

    with VectorStore(config) as store:
        if input_path:
            files = [input_path]
        elif input_dir:
            files = list(input_dir.glob("*_chunks.json"))
        else:
            raise ValueError("Either input_path or input_dir must be provided")

        if not files:
            logger.warning("No chunk files found")
            return

        logger.info(f"Ingesting {len(files)} chunk files...")

        success = 0
        errors = 0

        for file_path in files:
            try:
                video = load_chunked_video(file_path)
                store.insert_video(video)
                success += 1
            except Exception as e:
                logger.error(f"Error ingesting {file_path}: {e}")
                errors += 1

        # Run quantization after all inserts
        if success > 0 and config.quantize_after_insert:
            try:
                store.quantize()
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")

        logger.info(f"Ingestion complete: {success} succeeded, {errors} failed")

        # Print stats
        stats = store.get_stats()
        logger.info(f"Store stats: {stats}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    """CLI entry point for the vector store."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Vector store for YouTube transcript chunks"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest chunks into the store")
    ingest_parser.add_argument("--input", type=Path, help="Single chunk JSON file")
    ingest_parser.add_argument("--input-dir", type=Path, help="Directory of chunk files")
    ingest_parser.add_argument(
        "--db", type=Path, default=Path("data/vector_store.db"),
        help="Database path"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar chunks")
    search_parser.add_argument("query", help="Search query text")
    search_parser.add_argument("-k", type=int, default=5, help="Number of results")
    search_parser.add_argument(
        "--db", type=Path, default=Path("data/vector_store.db"),
        help="Database path"
    )
    search_parser.add_argument(
        "--with-context", action="store_true",
        help="Include parent chunk context"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show store statistics")
    stats_parser.add_argument(
        "--db", type=Path, default=Path("data/vector_store.db"),
        help="Database path"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List all videos")
    list_parser.add_argument(
        "--db", type=Path, default=Path("data/vector_store.db"),
        help="Database path"
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a video")
    delete_parser.add_argument("video_id", help="Video ID to delete")
    delete_parser.add_argument(
        "--db", type=Path, default=Path("data/vector_store.db"),
        help="Database path"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_chunks(
            input_path=args.input,
            input_dir=args.input_dir,
            db_path=args.db,
        )

    elif args.command == "search":
        config = VectorStoreConfig(db_path=args.db)
        with VectorStore(config) as store:
            if args.with_context:
                results = store.search_with_context(args.query, k=args.k)
                print(f"\nSearch: \"{args.query}\"\n")
                for i, r in enumerate(results, 1):
                    mins = int(r.chunk.start_time // 60)
                    secs = int(r.chunk.start_time % 60)
                    print(f"{i}. [{r.chunk.distance:.3f}] {r.chunk.video_title} ({mins}:{secs:02d})")
                    print(f"   {r.chunk.youtube_link}")
                    print(f"   \"{r.chunk.text[:200]}...\"")
                    if r.parent_chunk_id:
                        print(f"   [Parent context: {len(r.parent_text)} chars]")
                    print()
            else:
                results = store.search(args.query, k=args.k)
                print(f"\nSearch: \"{args.query}\"\n")
                for i, r in enumerate(results, 1):
                    mins = int(r.start_time // 60)
                    secs = int(r.start_time % 60)
                    print(f"{i}. [{r.distance:.3f}] {r.video_title} ({mins}:{secs:02d})")
                    print(f"   {r.youtube_link}")
                    print(f"   \"{r.text[:200]}...\"")
                    print()

    elif args.command == "stats":
        config = VectorStoreConfig(db_path=args.db)
        with VectorStore(config) as store:
            stats = store.get_stats()
            print("\nVector Store Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    elif args.command == "list":
        config = VectorStoreConfig(db_path=args.db)
        with VectorStore(config) as store:
            videos = store.list_videos()
            print(f"\nVideos in store: {len(videos)}\n")
            for v in videos:
                print(f"  {v['video_id']}")
                print(f"    Title: {v['video_title']}")
                print(f"    Chunks: {v['total_chunks']} (parents: {v['total_parent_chunks']})")
                print()

    elif args.command == "delete":
        config = VectorStoreConfig(db_path=args.db)
        with VectorStore(config) as store:
            if store.delete_video(args.video_id):
                print(f"Deleted video: {args.video_id}")
            else:
                print(f"Video not found: {args.video_id}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
