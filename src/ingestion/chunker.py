"""
Hybrid Time-Semantic Chunker for YouTube transcripts.

Splits preprocessed transcripts into chunks optimized for retrieval:
- Semantically coherent (complete thoughts) using embedding similarity
- Appropriately sized for BGE-M3 (~350 tokens)
- Preserving timestamp ranges for deep-linking
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for chunking parameters."""
    target_tokens: int = 350
    min_tokens: int = 150
    max_tokens: int = 450
    hard_max_tokens: int = 500
    overlap_tokens: int = 50
    pause_threshold_seconds: float = 3.0
    similarity_threshold: float = 0.7  # Boundaries where similarity < this
    max_continuation_overflow: int = 100
    embedding_strategy: str = "local"  # "local" or "deepinfra"


@dataclass
class Sentence:
    """A sentence with its timestamp range (from preprocessor)."""
    text: str
    start: float
    end: float


@dataclass
class Chunk:
    """A chunk of text optimized for retrieval."""
    chunk_id: str
    chunk_index: int
    video_id: str
    video_title: str
    channel: str
    text: str
    start_time: float
    end_time: float
    token_count: int
    language: str
    context_before: str = ""
    context_after: str = ""
    youtube_link: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ChunkedVideo:
    """Chunked data for a single video."""
    video_id: str
    video_title: str
    channel: str
    language: str
    total_chunks: int
    chunks: list[Chunk]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "channel": self.channel,
            "language": self.language,
            "total_chunks": self.total_chunks,
            "chunks": [c.to_dict() for c in self.chunks],
        }


class HybridChunker:
    """
    Hybrid time-semantic chunker for YouTube transcripts.

    Uses embedding-based boundary detection (BGE-M3) combined with
    temporal pause detection to find natural chunk boundaries.
    """

    # Continuation patterns - sentences starting with these should stay with previous
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

    def __init__(
        self,
        config: ChunkConfig | None = None,
        embedding_strategy: "EmbeddingStrategy | None" = None,
    ):
        """
        Initialize the chunker with configuration.

        Args:
            config: Chunking configuration
            embedding_strategy: Optional embedding strategy instance.
                If not provided, creates one based on config.embedding_strategy.
        """
        self.config = config or ChunkConfig()
        self._compiled_continuation_patterns = self._compile_continuation_patterns()

        # Use injected strategy or create from config
        if embedding_strategy is not None:
            self._embedding_strategy = embedding_strategy
        else:
            from .embeddings import create_embedding_strategy
            self._embedding_strategy = create_embedding_strategy(
                self.config.embedding_strategy
            )

    def _compile_continuation_patterns(self) -> list[re.Pattern]:
        """Compile continuation patterns."""
        return [re.compile(p, re.IGNORECASE) for p in self.CONTINUATION_PATTERNS]

    def compute_sentence_embeddings(self, sentences: list[Sentence]) -> np.ndarray:
        """
        Compute embeddings for all sentences using the configured strategy.

        Args:
            sentences: List of sentences to embed

        Returns:
            Numpy array of shape (n_sentences, embedding_dim)
        """
        if not sentences:
            return np.array([])

        texts = [s.text for s in sentences]
        logger.debug(f"Computing embeddings for {len(texts)} sentences")
        return self._embedding_strategy.embed(texts)

    def detect_boundaries_by_embedding(
        self,
        sentences: list[Sentence],
        embeddings: np.ndarray,
    ) -> list[int]:
        """
        Detect boundary indices using cosine similarity drops.

        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings array

        Returns:
            List of sentence indices where boundaries occur
            (i.e., the next chunk should start at this index)
        """
        if len(sentences) < 2:
            return []

        boundaries = []

        for i in range(1, len(sentences)):
            # Cosine similarity between consecutive sentences
            norm_prev = np.linalg.norm(embeddings[i - 1])
            norm_curr = np.linalg.norm(embeddings[i])

            if norm_prev > 0 and norm_curr > 0:
                similarity = np.dot(embeddings[i - 1], embeddings[i]) / (norm_prev * norm_curr)
            else:
                similarity = 0.0

            if similarity < self.config.similarity_threshold:
                boundaries.append(i)
                logger.debug(
                    f"Embedding boundary at sentence {i}: similarity={similarity:.3f}"
                )

        return boundaries

    def detect_boundaries_by_pause(self, sentences: list[Sentence]) -> list[int]:
        """
        Detect boundaries where there are long pauses (> threshold).

        Args:
            sentences: List of sentences with timestamps

        Returns:
            List of sentence indices where boundaries occur
        """
        if len(sentences) < 2:
            return []

        boundaries = []

        for i in range(1, len(sentences)):
            gap = sentences[i].start - sentences[i - 1].end
            if gap > self.config.pause_threshold_seconds:
                boundaries.append(i)
                logger.debug(
                    f"Pause boundary at sentence {i}: gap={gap:.1f}s"
                )

        return boundaries

    def detect_boundaries(self, sentences: list[Sentence]) -> list[int]:
        """
        Detect chunk boundaries using combined signals:
        1. Embedding similarity drops
        2. Temporal pauses

        Args:
            sentences: List of sentences

        Returns:
            Sorted list of unique boundary indices
        """
        if len(sentences) < 2:
            return []

        # Signal 1: Embedding-based
        embeddings = self.compute_sentence_embeddings(sentences)
        embedding_boundaries = self.detect_boundaries_by_embedding(sentences, embeddings)
        logger.info(f"Found {len(embedding_boundaries)} embedding-based boundaries")

        # Signal 2: Temporal gaps
        pause_boundaries = self.detect_boundaries_by_pause(sentences)
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

    def is_continuation(self, sentence: Sentence) -> bool:
        """Check if sentence should stay with previous chunk."""
        for pattern in self._compiled_continuation_patterns:
            if pattern.match(sentence.text):
                return True
        return False

    def create_chunk(
        self,
        sentences: list[Sentence],
        index: int,
        video_id: str,
        video_title: str,
        channel: str,
        language: str,
    ) -> Chunk:
        """Create a chunk from a list of sentences."""
        text = " ".join(s.text for s in sentences)
        start_time = sentences[0].start
        end_time = sentences[-1].end
        token_count = self.count_tokens(text)

        # Generate YouTube link (time in integer seconds)
        youtube_link = f"https://youtube.com/watch?v={video_id}&t={int(start_time)}"
        chunk_id = f"{video_id}_{index:03d}"

        return Chunk(
            chunk_id=chunk_id,
            chunk_index=index,
            video_id=video_id,
            video_title=video_title,
            channel=channel,
            text=text,
            start_time=start_time,
            end_time=end_time,
            token_count=token_count,
            language=language,
            youtube_link=youtube_link,
        )

    def merge_chunks(self, chunk: Chunk, sentences: list[Sentence]) -> Chunk:
        """Merge additional sentences into an existing chunk."""
        additional_text = " ".join(s.text for s in sentences)
        chunk.text = chunk.text + " " + additional_text
        chunk.end_time = sentences[-1].end
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

    def _chunk_segment(
        self,
        sentences: list[Sentence],
        chunk_index: int,
        video_id: str,
        video_title: str,
        channel: str,
        language: str,
    ) -> tuple[list[Chunk], int]:
        """
        Create chunks from a segment of sentences respecting token limits.

        Args:
            sentences: Segment of sentences to chunk
            chunk_index: Starting chunk index
            video_id, video_title, channel, language: Metadata

        Returns:
            Tuple of (list of chunks, next chunk index)
        """
        chunks = []
        current_chunk_sentences: list[Sentence] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence.text)
            would_exceed = current_tokens + sentence_tokens > self.config.max_tokens

            if would_exceed and current_tokens >= self.config.min_tokens:
                # Check for continuation patterns before splitting
                can_continue = (
                    self.is_continuation(sentence) and
                    current_tokens + sentence_tokens <= self.config.hard_max_tokens
                )

                if can_continue:
                    current_chunk_sentences.append(sentence)
                    current_tokens += sentence_tokens
                    continue

                # Finalize current chunk
                chunk = self.create_chunk(
                    sentences=current_chunk_sentences,
                    index=chunk_index,
                    video_id=video_id,
                    video_title=video_title,
                    channel=channel,
                    language=language,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Reset for next chunk
                current_chunk_sentences = []
                current_tokens = 0

            current_chunk_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Handle remaining sentences
        if current_chunk_sentences:
            if current_tokens >= self.config.min_tokens:
                chunk = self.create_chunk(
                    sentences=current_chunk_sentences,
                    index=chunk_index,
                    video_id=video_id,
                    video_title=video_title,
                    channel=channel,
                    language=language,
                )
                chunks.append(chunk)
                chunk_index += 1
            elif chunks:
                # Merge with previous chunk if too small
                chunks[-1] = self.merge_chunks(chunks[-1], current_chunk_sentences)

        return chunks, chunk_index

    def chunk_transcript(
        self,
        processed_transcript: dict[str, Any],
        video_title: str = "",
        channel: str = "",
    ) -> ChunkedVideo:
        """
        Main chunking algorithm using embedding-based boundary detection.

        1. Detect boundaries using embedding similarity + pause detection
        2. Create chunks between boundaries respecting token limits
        3. Apply overlap for context preservation

        Args:
            processed_transcript: Processed transcript from preprocessor
            video_title: Video title (optional)
            channel: Channel name (optional)

        Returns:
            ChunkedVideo with all chunks
        """
        video_id = processed_transcript.get("video_id", "unknown")
        language = processed_transcript.get("language", "unknown")
        video_title = video_title or processed_transcript.get("video_title", "Unknown")

        # Convert dict sentences to Sentence objects
        sentences = [
            Sentence(text=s["text"], start=s["start"], end=s["end"])
            for s in processed_transcript.get("sentences", [])
        ]

        if not sentences:
            logger.warning(f"No sentences in transcript for video {video_id}")
            return ChunkedVideo(
                video_id=video_id,
                video_title=video_title,
                channel=channel,
                language=language,
                total_chunks=0,
                chunks=[],
            )

        logger.info(f"Chunking video {video_id} with {len(sentences)} sentences")

        # Step 1: Detect boundaries using embeddings + pauses
        boundaries = self.detect_boundaries(sentences)
        logger.info(f"Detected {len(boundaries)} total boundaries")

        # Step 2: Create chunks between boundaries
        all_chunks: list[Chunk] = []
        chunk_index = 0
        start_idx = 0

        # Add end of transcript as final boundary
        boundary_indices = boundaries + [len(sentences)]

        for boundary_idx in boundary_indices:
            segment_sentences = sentences[start_idx:boundary_idx]

            if segment_sentences:
                segment_chunks, chunk_index = self._chunk_segment(
                    sentences=segment_sentences,
                    chunk_index=chunk_index,
                    video_id=video_id,
                    video_title=video_title,
                    channel=channel,
                    language=language,
                )
                all_chunks.extend(segment_chunks)

            start_idx = boundary_idx

        # Step 3: Add overlap context
        self.add_overlap_context(all_chunks)

        logger.info(f"Created {len(all_chunks)} chunks for video {video_id}")

        # Log chunk statistics
        if all_chunks:
            token_counts = [c.token_count for c in all_chunks]
            avg_tokens = sum(token_counts) / len(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            logger.info(
                f"Chunk stats - avg: {avg_tokens:.0f}, min: {min_tokens}, max: {max_tokens} tokens"
            )

        return ChunkedVideo(
            video_id=video_id,
            video_title=video_title,
            channel=channel,
            language=language,
            total_chunks=len(all_chunks),
            chunks=all_chunks,
        )

    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        video_title: str = "",
        channel: str = "",
    ) -> ChunkedVideo:
        """
        Process a single processed transcript file.

        Args:
            input_path: Path to processed transcript JSON
            output_path: Path to save chunked output
            video_title: Video title (optional)
            channel: Channel name (optional)

        Returns:
            ChunkedVideo object
        """
        logger.info(f"Processing file: {input_path}")

        # Load processed transcript
        processed_transcript = json.loads(input_path.read_text(encoding="utf-8"))

        # Try to get video metadata from raw transcript if available
        raw_transcript_path = input_path.parent.parent / "raw_transcripts" / input_path.name
        if raw_transcript_path.exists() and (not video_title or not channel):
            raw_data = json.loads(raw_transcript_path.read_text(encoding="utf-8"))
            video_title = video_title or raw_data.get("title", "")
            channel = channel or raw_data.get("channel", "")

        # Chunk the transcript
        result = self.chunk_transcript(
            processed_transcript,
            video_title=video_title,
            channel=channel,
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
        description="Chunk preprocessed transcripts for RAG pipeline"
    )
    parser.add_argument(
        "--video-id",
        help="Process a single video by ID",
    )
    parser.add_argument(
        "--input-dir",
        default="data/processed_transcripts",
        help="Input directory containing processed transcripts",
    )
    parser.add_argument(
        "--output-dir",
        default="data/chunks",
        help="Output directory for chunked data",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all processed transcripts in input directory",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=350,
        help="Target token count per chunk (default: 350)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=150,
        help="Minimum token count per chunk (default: 150)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=450,
        help="Maximum token count per chunk (default: 450)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for boundary detection (default: 0.7)",
    )
    parser.add_argument(
        "--embedding-strategy",
        choices=["local", "deepinfra"],
        default="local",
        help="Embedding strategy: local (GPU/CPU) or deepinfra (API)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ChunkConfig(
        target_tokens=args.target_tokens,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        similarity_threshold=args.similarity_threshold,
        embedding_strategy=args.embedding_strategy,
    )

    chunker = HybridChunker(config=config)

    if args.video_id:
        # Process single video
        input_path = input_dir / f"{args.video_id}.json"
        output_path = output_dir / f"{args.video_id}_chunks.json"

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        chunker.process_file(input_path, output_path)

    elif args.batch:
        # Process all transcripts
        input_files = list(input_dir.glob("*.json"))

        if not input_files:
            logger.warning(f"No transcript files found in {input_dir}")
            return

        logger.info(f"Processing {len(input_files)} transcripts")

        success_count = 0
        error_count = 0
        total_chunks = 0

        # Also create a manifest
        manifest = {"videos": {}, "total_chunks": 0}

        for input_path in input_files:
            video_id = input_path.stem
            output_path = output_dir / f"{video_id}_chunks.json"

            try:
                result = chunker.process_file(input_path, output_path)
                success_count += 1
                total_chunks += result.total_chunks

                manifest["videos"][video_id] = {
                    "title": result.video_title,
                    "chunks": result.total_chunks,
                    "language": result.language,
                }
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                error_count += 1

        manifest["total_chunks"] = total_chunks

        # Save manifest
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(
            f"Processing complete. Success: {success_count}, Errors: {error_count}, "
            f"Total chunks: {total_chunks}"
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
