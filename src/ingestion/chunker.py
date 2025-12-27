"""
Hybrid Chunker for YouTube transcripts using Gemini embeddings.

Processes Gemini transcription output directly and creates chunks
optimized for RAG retrieval with semantic boundary detection.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for chunking parameters."""
    # Child chunk settings (for embedding/search)
    child_target_tokens: int = 200
    child_min_tokens: int = 128
    child_max_tokens: int = 256
    child_hard_max_tokens: int = 300

    # Parent chunk settings (for LLM context)
    parent_target_tokens: int = 800
    parent_min_tokens: int = 512
    parent_max_tokens: int = 1000

    # Enable parent chunking
    enable_parent_chunks: bool = True

    # Shared settings
    overlap_tokens: int = 50
    pause_threshold_seconds: float = 3.0
    similarity_threshold: float = 0.5  # Lower threshold = fewer boundaries
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768  # Recommended dimensions for efficiency


@dataclass
class Segment:
    """A segment from Gemini transcription output."""
    text: str
    start: float
    end: float
    speaker: str = ""
    language: str = ""


@dataclass
class Chunk:
    """A child chunk of text optimized for retrieval (embedded for search)."""
    chunk_id: str
    chunk_index: int
    video_id: str
    video_title: str
    text: str
    start_time: float
    end_time: float
    token_count: int
    section_type: str = "main_content"
    context_before: str = ""
    context_after: str = ""
    youtube_link: str = ""
    parent_chunk_id: str = ""  # Reference to parent chunk for context expansion
    youtube_video_id: str = ""  # Actual YouTube video ID for URL generation

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ParentChunk:
    """A parent chunk providing expanded context for LLM (NOT embedded)."""
    parent_chunk_id: str
    parent_index: int
    video_id: str
    video_title: str
    text: str
    start_time: float
    end_time: float
    token_count: int
    child_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ChunkedVideo:
    """Chunked data for a single video with parent-child hierarchy."""
    video_id: str
    video_title: str
    total_chunks: int
    chunks: list[Chunk]  # Child chunks (embedded for search)
    total_parent_chunks: int = 0
    parent_chunks: list[ParentChunk] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "total_chunks": self.total_chunks,
            "chunks": [c.to_dict() for c in self.chunks],
        }
        if self.parent_chunks:
            result["total_parent_chunks"] = self.total_parent_chunks
            result["parent_chunks"] = [p.to_dict() for p in self.parent_chunks]
        return result


class GeminiEmbedding:
    """Embedding strategy using Google Gemini API."""

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        task_type: str = "SEMANTIC_SIMILARITY",
    ):
        """
        Initialize Gemini embedding client.

        Args:
            model: Embedding model name
            dimensions: Output embedding dimensions (128-3072, recommended: 768, 1536, 3072)
            task_type: Task type for embeddings. Options:
                - SEMANTIC_SIMILARITY: Compare text similarity
                - RETRIEVAL_DOCUMENT: For documents in retrieval
                - RETRIEVAL_QUERY: For queries in retrieval
                - CLASSIFICATION, CLUSTERING, etc.
        """
        self.model = model
        self.dimensions = dimensions
        self.task_type = task_type
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        """Lazy-load the GenAI client."""
        if self._client is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def embed(self, texts: list[str], max_retries: int = 3) -> np.ndarray:
        """
        Compute embeddings for a list of texts with rate limiting.

        Args:
            texts: List of texts to embed
            max_retries: Maximum retry attempts for rate limit errors

        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        client = self._get_client()
        logger.debug(f"Computing embeddings for {len(texts)} texts")

        # Gemini API can handle batches
        embeddings = []
        batch_size = 100  # API limit

        config = types.EmbedContentConfig(
            task_type=self.task_type,
            output_dimensionality=self.dimensions,
        )

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Retry logic for rate limits
            for attempt in range(max_retries):
                try:
                    result = client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=config,
                    )
                    for embedding in result.embeddings:
                        embeddings.append(embedding.values)
                    break  # Success, exit retry loop
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait_time = 2 ** attempt * 10  # 10s, 20s, 40s
                        logger.warning(
                            f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        if attempt == max_retries - 1:
                            raise
                    else:
                        raise

            # Small delay between batches to avoid rate limits
            if i + batch_size < len(texts):
                time.sleep(0.5)

        return np.array(embeddings)


class HybridChunker:
    """
    Hybrid chunker for YouTube transcripts.

    Uses Gemini embeddings for semantic boundary detection combined
    with temporal pause detection to find natural chunk boundaries.
    """

    # Continuation patterns - segments starting with these should stay with previous
    CONTINUATION_PATTERNS = [
        # Examples
        r"^(Par exemple|For example|For instance|Prenons)",
        # Clarification
        r"^(C'est-à-dire|That is|In other words|Autrement dit)",
        # Consequence
        r"^(Donc|So|Thus|Therefore|Alors)",
        # Addition
        r"^(Et |And |De plus|Furthermore|Also|En plus)",
        # Contrast
        r"^(Mais|But|However|Cependant|Toutefois)",
        # Cause
        r"^(Parce que|Because|Car|Since|Puisque)",
        # Reference
        r"^(Ce qui|Which|That|Cela|Ça)",
        # Lists
        r"^\d+\.",  # Numbered list continuation
        r"^(Deuxièmement|Troisièmement|Second|Third|Ensuite|Puis)",
        # Demonstration
        r"^(Ici|Here|Voici|Look|Regardez|On voit)",
        # Continuation markers
        r"^(En fait|Actually|D'ailleurs|Moreover)",
        # Code/tech explanations
        r"^(Cette|This|Ces|These)",
    ]

    def __init__(self, config: ChunkConfig | None = None):
        """
        Initialize the chunker with configuration.

        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkConfig()
        self._compiled_continuation_patterns = self._compile_continuation_patterns()
        self._embedding_client: GeminiEmbedding | None = None

    def _get_embedding_client(self) -> GeminiEmbedding:
        """Lazy-load the embedding client."""
        if self._embedding_client is None:
            self._embedding_client = GeminiEmbedding(
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dimensions,
                task_type="SEMANTIC_SIMILARITY",
            )
        return self._embedding_client

    def _compile_continuation_patterns(self) -> list[re.Pattern]:
        """Compile continuation patterns."""
        return [re.compile(p, re.IGNORECASE) for p in self.CONTINUATION_PATTERNS]

    @staticmethod
    def parse_timestamp(timestamp: str) -> float:
        """
        Parse Gemini timestamp format to float seconds.

        Handles formats:
        - MM:SS:mmm -> seconds (e.g., "00:06:582" -> 6.582)
        - HH:MM:SS:mmm -> seconds
        - HH:MM:SS,mmm -> seconds (SRT format)

        Args:
            timestamp: Timestamp string

        Returns:
            Time in seconds as float
        """
        # Replace comma with colon for uniform parsing
        ts = timestamp.replace(",", ":")
        parts = ts.split(":")

        if len(parts) == 3:
            # Format: MM:SS:mmm
            mm, ss, mmm = parts
            return int(mm) * 60 + int(ss) + int(mmm) / 1000
        elif len(parts) == 4:
            # Format: HH:MM:SS:mmm
            hh, mm, ss, mmm = parts
            return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(mmm) / 1000
        else:
            logger.warning(f"Unknown timestamp format: {timestamp}")
            return 0.0

    def load_transcript(self, transcript_path: Path) -> list[Segment]:
        """
        Load and parse a Gemini transcription JSON file.

        Args:
            transcript_path: Path to the transcription JSON

        Returns:
            List of Segment objects (excludes empty segments)
        """
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        segments = []

        for seg in data.get("segments", []):
            text = seg.get("content", "").strip()
            # Skip empty segments
            if not text:
                continue

            start = self.parse_timestamp(seg.get("start_time", "00:00:000"))
            end = self.parse_timestamp(seg.get("end_time", "00:00:000"))

            segments.append(Segment(
                text=text,
                start=start,
                end=end,
                speaker=seg.get("speaker", ""),
                language=seg.get("language", ""),
            ))

        return segments

    def compute_segment_embeddings(self, segments: list[Segment]) -> np.ndarray:
        """
        Compute embeddings for all segments.

        Args:
            segments: List of segments to embed

        Returns:
            Numpy array of shape (n_segments, embedding_dim)
        """
        if not segments:
            return np.array([])

        texts = [s.text for s in segments]
        return self._get_embedding_client().embed(texts)

    def detect_boundaries_by_embedding(
        self,
        segments: list[Segment],
        embeddings: np.ndarray,
    ) -> list[int]:
        """
        Detect boundary indices using cosine similarity drops.

        Args:
            segments: List of segments
            embeddings: Segment embeddings array

        Returns:
            List of segment indices where boundaries occur
        """
        if len(segments) < 2:
            return []

        boundaries = []

        for i in range(1, len(segments)):
            # Cosine similarity between consecutive segments
            norm_prev = np.linalg.norm(embeddings[i - 1])
            norm_curr = np.linalg.norm(embeddings[i])

            if norm_prev > 0 and norm_curr > 0:
                similarity = np.dot(embeddings[i - 1], embeddings[i]) / (norm_prev * norm_curr)
            else:
                similarity = 0.0

            if similarity < self.config.similarity_threshold:
                boundaries.append(i)
                logger.debug(
                    f"Embedding boundary at segment {i}: similarity={similarity:.3f}"
                )

        return boundaries

    def detect_boundaries_by_pause(self, segments: list[Segment]) -> list[int]:
        """
        Detect boundaries where there are long pauses.

        Args:
            segments: List of segments with timestamps

        Returns:
            List of segment indices where boundaries occur
        """
        if len(segments) < 2:
            return []

        boundaries = []

        for i in range(1, len(segments)):
            gap = segments[i].start - segments[i - 1].end
            if gap > self.config.pause_threshold_seconds:
                boundaries.append(i)
                logger.debug(f"Pause boundary at segment {i}: gap={gap:.1f}s")

        return boundaries

    def detect_boundaries(self, segments: list[Segment]) -> list[int]:
        """
        Detect chunk boundaries using combined signals.

        Args:
            segments: List of segments

        Returns:
            Sorted list of unique boundary indices
        """
        if len(segments) < 2:
            return []

        # Signal 1: Embedding-based
        logger.info("Computing embeddings for boundary detection...")
        embeddings = self.compute_segment_embeddings(segments)
        embedding_boundaries = self.detect_boundaries_by_embedding(segments, embeddings)
        logger.info(f"Found {len(embedding_boundaries)} embedding-based boundaries")

        # Signal 2: Temporal gaps
        pause_boundaries = self.detect_boundaries_by_pause(segments)
        logger.info(f"Found {len(pause_boundaries)} pause-based boundaries")

        # Combine and dedupe
        all_boundaries = set(embedding_boundaries) | set(pause_boundaries)
        return sorted(all_boundaries)

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses word count * 1.3 as a heuristic for multilingual text.
        """
        if not text:
            return 0
        words = text.split()
        return int(len(words) * 1.3)

    def is_continuation(self, segment: Segment) -> bool:
        """Check if segment should stay with previous chunk."""
        for pattern in self._compiled_continuation_patterns:
            if pattern.match(segment.text):
                return True
        return False

    def create_chunk(
        self,
        segments: list[Segment],
        index: int,
        video_id: str,
        video_title: str,
        youtube_video_id: str = "",
    ) -> Chunk:
        """Create a chunk from a list of segments."""
        text = " ".join(s.text for s in segments)
        start_time = segments[0].start
        end_time = segments[-1].end
        token_count = self.count_tokens(text)

        # Generate YouTube link (time in integer seconds)
        # Use youtube_video_id if provided, otherwise fall back to video_id
        yt_id = youtube_video_id or video_id
        youtube_link = f"https://www.youtube.com/watch?v={yt_id}&t={int(start_time)}s"
        chunk_id = f"{video_id}_{index:03d}"

        return Chunk(
            chunk_id=chunk_id,
            chunk_index=index,
            video_id=video_id,
            video_title=video_title,
            text=text,
            start_time=start_time,
            end_time=end_time,
            token_count=token_count,
            section_type="main_content",
            youtube_link=youtube_link,
            youtube_video_id=youtube_video_id,
        )

    def merge_chunks(self, chunk: Chunk, segments: list[Segment]) -> Chunk:
        """Merge additional segments into an existing chunk."""
        additional_text = " ".join(s.text for s in segments)
        chunk.text = chunk.text + " " + additional_text
        chunk.end_time = segments[-1].end
        chunk.token_count = self.count_tokens(chunk.text)
        return chunk

    def get_last_n_tokens(self, text: str, n_tokens: int) -> str:
        """Get approximately the last n tokens of text."""
        if not text:
            return ""
        words = text.split()
        n_words = max(1, int(n_tokens / 1.3))
        if len(words) <= n_words:
            return text
        return " ".join(words[-n_words:])

    def get_first_n_tokens(self, text: str, n_tokens: int) -> str:
        """Get approximately the first n tokens of text."""
        if not text:
            return ""
        words = text.split()
        n_words = max(1, int(n_tokens / 1.3))
        if len(words) <= n_words:
            return text
        return " ".join(words[:n_words])

    def add_overlap_context(self, chunks: list[Chunk]) -> None:
        """Add context_before and context_after to each chunk."""
        for i, chunk in enumerate(chunks):
            if i > 0:
                chunk.context_before = self.get_last_n_tokens(
                    chunks[i - 1].text, self.config.overlap_tokens
                )
            if i < len(chunks) - 1:
                chunk.context_after = self.get_first_n_tokens(
                    chunks[i + 1].text, self.config.overlap_tokens
                )

    def _chunk_segment_group(
        self,
        segments: list[Segment],
        chunk_index: int,
        video_id: str,
        video_title: str,
    ) -> tuple[list[Chunk], int]:
        """
        Create child chunks from a group of segments respecting token limits.

        Args:
            segments: Group of segments to chunk
            chunk_index: Starting chunk index
            video_id, video_title: Metadata

        Returns:
            Tuple of (list of chunks, next chunk index)
        """
        chunks = []
        current_chunk_segments: list[Segment] = []
        current_tokens = 0

        for segment in segments:
            segment_tokens = self.count_tokens(segment.text)
            would_exceed = current_tokens + segment_tokens > self.config.child_max_tokens

            if would_exceed and current_tokens >= self.config.child_min_tokens:
                # Check for continuation patterns before splitting
                can_continue = (
                    self.is_continuation(segment) and
                    current_tokens + segment_tokens <= self.config.child_hard_max_tokens
                )

                if can_continue:
                    current_chunk_segments.append(segment)
                    current_tokens += segment_tokens
                    continue

                # Finalize current chunk
                chunk = self.create_chunk(
                    segments=current_chunk_segments,
                    index=chunk_index,
                    video_id=video_id,
                    video_title=video_title,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Reset for next chunk
                current_chunk_segments = []
                current_tokens = 0

            current_chunk_segments.append(segment)
            current_tokens += segment_tokens

        # Handle remaining segments
        if current_chunk_segments:
            if current_tokens >= self.config.child_min_tokens:
                chunk = self.create_chunk(
                    segments=current_chunk_segments,
                    index=chunk_index,
                    video_id=video_id,
                    video_title=video_title,
                )
                chunks.append(chunk)
                chunk_index += 1
            elif chunks:
                # Merge with previous chunk if too small
                chunks[-1] = self.merge_chunks(chunks[-1], current_chunk_segments)

        return chunks, chunk_index

    def group_into_parents(
        self,
        chunks: list[Chunk],
        video_id: str,
        video_title: str,
    ) -> list[ParentChunk]:
        """
        Group child chunks into parent chunks for expanded LLM context.

        Args:
            chunks: List of child chunks to group
            video_id: Video identifier
            video_title: Video title

        Returns:
            List of parent chunks with child references
        """
        if not chunks:
            return []

        parent_chunks: list[ParentChunk] = []
        current_children: list[Chunk] = []
        current_tokens = 0
        parent_index = 0

        for chunk in chunks:
            would_exceed = current_tokens + chunk.token_count > self.config.parent_max_tokens
            reached_target = current_tokens >= self.config.parent_target_tokens

            # Split if we'd exceed max, or we've reached target with enough children
            should_split = False
            if current_children:
                if would_exceed and current_tokens >= self.config.parent_min_tokens:
                    should_split = True
                elif reached_target and len(current_children) >= 2:
                    # Prefer splitting at target if we have multiple children
                    should_split = True

            if should_split:
                # Create parent from accumulated children
                parent = self._create_parent_chunk(
                    children=current_children,
                    parent_index=parent_index,
                    video_id=video_id,
                    video_title=video_title,
                )
                parent_chunks.append(parent)
                parent_index += 1
                current_children = []
                current_tokens = 0

            current_children.append(chunk)
            current_tokens += chunk.token_count

        # Handle remaining children
        if current_children:
            if current_tokens >= self.config.parent_min_tokens or not parent_chunks:
                parent = self._create_parent_chunk(
                    children=current_children,
                    parent_index=parent_index,
                    video_id=video_id,
                    video_title=video_title,
                )
                parent_chunks.append(parent)
            elif parent_chunks:
                # Merge with previous parent if too small
                last_parent = parent_chunks[-1]
                for child in current_children:
                    child.parent_chunk_id = last_parent.parent_chunk_id
                    last_parent.child_chunk_ids.append(child.chunk_id)
                last_parent.text += " " + " ".join(c.text for c in current_children)
                last_parent.end_time = current_children[-1].end_time
                last_parent.token_count = self.count_tokens(last_parent.text)

        return parent_chunks

    def _create_parent_chunk(
        self,
        children: list[Chunk],
        parent_index: int,
        video_id: str,
        video_title: str,
    ) -> ParentChunk:
        """Create a parent chunk from a list of child chunks."""
        parent_chunk_id = f"{video_id}_P{parent_index:03d}"

        # Assign parent reference to all children
        for child in children:
            child.parent_chunk_id = parent_chunk_id

        # Combine text from all children
        combined_text = " ".join(c.text for c in children)

        return ParentChunk(
            parent_chunk_id=parent_chunk_id,
            parent_index=parent_index,
            video_id=video_id,
            video_title=video_title,
            text=combined_text,
            start_time=children[0].start_time,
            end_time=children[-1].end_time,
            token_count=self.count_tokens(combined_text),
            child_chunk_ids=[c.chunk_id for c in children],
        )

    def chunk_transcript(
        self,
        segments: list[Segment],
        video_id: str,
        video_title: str = "",
        youtube_video_id: str = "",
    ) -> ChunkedVideo:
        """
        Main chunking algorithm using embedding-based boundary detection.

        Creates child chunks (128-256 tokens) for embedding/search, and optionally
        groups them into parent chunks (512-1000 tokens) for expanded LLM context.

        Strategy: Accumulate segments until target tokens reached, then prefer
        splitting at detected boundaries. Boundaries are "soft" suggestions,
        not hard requirements.

        Args:
            segments: List of transcript segments
            video_id: Video identifier
            video_title: Video title (optional)
            youtube_video_id: Actual YouTube video ID for URL generation (optional)

        Returns:
            ChunkedVideo with child chunks and optional parent chunks
        """
        if not segments:
            logger.warning(f"No segments in transcript for video {video_id}")
            return ChunkedVideo(
                video_id=video_id,
                video_title=video_title,
                total_chunks=0,
                chunks=[],
            )

        logger.info(f"Chunking video {video_id} with {len(segments)} segments")

        # Step 1: Detect boundaries using embeddings + pauses
        boundaries = self.detect_boundaries(segments)
        boundary_set = set(boundaries)
        logger.info(f"Detected {len(boundaries)} potential boundaries")

        # Step 2: Accumulate segments into child chunks
        all_chunks: list[Chunk] = []
        current_segments: list[Segment] = []
        current_tokens = 0
        chunk_index = 0

        for i, segment in enumerate(segments):
            segment_tokens = self.count_tokens(segment.text)

            # Check if we should split before this segment
            would_exceed = current_tokens + segment_tokens > self.config.child_max_tokens
            at_boundary = i in boundary_set
            reached_target = current_tokens >= self.config.child_target_tokens

            # Decide whether to split
            should_split = False
            if current_segments:  # Only split if we have content
                if would_exceed and current_tokens >= self.config.child_min_tokens:
                    # Would exceed max, must split (unless continuation)
                    if self.is_continuation(segment):
                        if current_tokens + segment_tokens <= self.config.child_hard_max_tokens:
                            # Allow continuation within hard max
                            should_split = False
                        else:
                            should_split = True
                    else:
                        should_split = True
                elif reached_target and at_boundary:
                    # At target and at a natural boundary - good place to split
                    should_split = True

            if should_split:
                # Create chunk from accumulated segments
                chunk = self.create_chunk(
                    segments=current_segments,
                    index=chunk_index,
                    video_id=video_id,
                    video_title=video_title,
                    youtube_video_id=youtube_video_id,
                )
                all_chunks.append(chunk)
                chunk_index += 1
                current_segments = []
                current_tokens = 0

            # Add current segment
            current_segments.append(segment)
            current_tokens += segment_tokens

        # Handle remaining segments
        if current_segments:
            if current_tokens >= self.config.child_min_tokens or not all_chunks:
                # Create final chunk
                chunk = self.create_chunk(
                    segments=current_segments,
                    index=chunk_index,
                    video_id=video_id,
                    video_title=video_title,
                    youtube_video_id=youtube_video_id,
                )
                all_chunks.append(chunk)
            else:
                # Merge with previous chunk
                all_chunks[-1] = self.merge_chunks(all_chunks[-1], current_segments)

        # Step 3: Add overlap context to child chunks
        self.add_overlap_context(all_chunks)

        logger.info(f"Created {len(all_chunks)} child chunks for video {video_id}")

        # Log child chunk statistics
        if all_chunks:
            token_counts = [c.token_count for c in all_chunks]
            avg_tokens = sum(token_counts) / len(token_counts)
            min_tok = min(token_counts)
            max_tok = max(token_counts)
            logger.info(
                f"Child chunk stats - avg: {avg_tokens:.0f}, min: {min_tok}, max: {max_tok} tokens"
            )

        # Step 4: Group into parent chunks if enabled
        parent_chunks: list[ParentChunk] = []
        if self.config.enable_parent_chunks:
            parent_chunks = self.group_into_parents(all_chunks, video_id, video_title)
            logger.info(f"Created {len(parent_chunks)} parent chunks")

            if parent_chunks:
                parent_token_counts = [p.token_count for p in parent_chunks]
                avg_parent = sum(parent_token_counts) / len(parent_token_counts)
                min_parent = min(parent_token_counts)
                max_parent = max(parent_token_counts)
                logger.info(
                    f"Parent chunk stats - avg: {avg_parent:.0f}, min: {min_parent}, max: {max_parent} tokens"
                )

        return ChunkedVideo(
            video_id=video_id,
            video_title=video_title,
            total_chunks=len(all_chunks),
            chunks=all_chunks,
            total_parent_chunks=len(parent_chunks),
            parent_chunks=parent_chunks,
        )

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        video_id: str | None = None,
        video_title: str | None = None,
    ) -> ChunkedVideo:
        """
        Process a single transcription file.

        Args:
            input_path: Path to Gemini transcription JSON
            output_path: Path to save chunked output
            video_id: Video ID (default: derive from filename)
            video_title: Video title (default: derive from filename)

        Returns:
            ChunkedVideo object
        """
        logger.info(f"Processing file: {input_path}")

        # Load transcript
        segments = self.load_transcript(input_path)

        # Derive metadata from filename if not provided
        if video_id is None:
            video_id = input_path.stem.replace(" ", "_")
        if video_title is None:
            # Convert filename to readable title
            video_title = input_path.stem.replace("_", " ").replace("-", " - ")

        # Chunk the transcript
        result = self.chunk_transcript(
            segments=segments,
            video_id=video_id,
            video_title=video_title,
        )

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Saved chunked transcript to: {output_path}")

        return result


def main():
    """CLI entry point for the hybrid chunker."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Chunk Gemini transcripts for RAG pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Input transcription JSON file",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/audio"),
        help="Input directory containing transcription JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/chunks"),
        help="Output directory for chunked data",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all transcripts in input directory",
    )
    # Child chunk settings
    parser.add_argument(
        "--child-target-tokens",
        type=int,
        default=200,
        help="Target token count per child chunk (default: 200)",
    )
    parser.add_argument(
        "--child-min-tokens",
        type=int,
        default=128,
        help="Minimum token count per child chunk (default: 128)",
    )
    parser.add_argument(
        "--child-max-tokens",
        type=int,
        default=256,
        help="Maximum token count per child chunk (default: 256)",
    )
    # Parent chunk settings
    parser.add_argument(
        "--parent-target-tokens",
        type=int,
        default=800,
        help="Target token count per parent chunk (default: 800)",
    )
    parser.add_argument(
        "--parent-min-tokens",
        type=int,
        default=512,
        help="Minimum token count per parent chunk (default: 512)",
    )
    parser.add_argument(
        "--parent-max-tokens",
        type=int,
        default=1000,
        help="Maximum token count per parent chunk (default: 1000)",
    )
    parser.add_argument(
        "--no-parent-chunks",
        action="store_true",
        help="Disable parent chunk generation (only create child chunks)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Cosine similarity threshold for boundary detection (default: 0.5)",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ChunkConfig(
        child_target_tokens=args.child_target_tokens,
        child_min_tokens=args.child_min_tokens,
        child_max_tokens=args.child_max_tokens,
        parent_target_tokens=args.parent_target_tokens,
        parent_min_tokens=args.parent_min_tokens,
        parent_max_tokens=args.parent_max_tokens,
        enable_parent_chunks=not args.no_parent_chunks,
        similarity_threshold=args.similarity_threshold,
    )

    chunker = HybridChunker(config=config)

    if args.input:
        # Process single file
        output_path = output_dir / f"{args.input.stem}_chunks.json"
        chunker.process_file(args.input, output_path)

    elif args.batch:
        # Process all transcripts
        input_dir = args.input_dir
        input_files = list(input_dir.glob("*.json"))

        if not input_files:
            logger.warning(f"No transcript files found in {input_dir}")
            return

        logger.info(f"Processing {len(input_files)} transcripts")

        success_count = 0
        error_count = 0
        total_chunks = 0
        total_parent_chunks = 0

        # Create manifest
        manifest = {"videos": {}, "total_chunks": 0, "total_parent_chunks": 0}

        for input_path in input_files:
            video_id = input_path.stem
            output_path = output_dir / f"{video_id}_chunks.json"

            try:
                result = chunker.process_file(input_path, output_path)
                success_count += 1
                total_chunks += result.total_chunks
                total_parent_chunks += result.total_parent_chunks

                manifest["videos"][video_id] = {
                    "title": result.video_title,
                    "chunks": result.total_chunks,
                    "parent_chunks": result.total_parent_chunks,
                }
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                error_count += 1

        manifest["total_chunks"] = total_chunks
        manifest["total_parent_chunks"] = total_parent_chunks

        # Save manifest
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"Processing complete. Success: {success_count}, Errors: {error_count}, "
            f"Total child chunks: {total_chunks}, Total parent chunks: {total_parent_chunks}"
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
