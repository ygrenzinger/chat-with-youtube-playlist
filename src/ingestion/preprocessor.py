"""
Transcript Preprocessor for YouTube transcripts.

Transforms raw YouTube transcript fragments into clean, punctuated,
sentence-structured text while preserving timestamp alignment.
"""

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from deepmultilingualpunctuation import PunctuationModel
from langdetect import detect, LangDetectException

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Sentence:
    """A sentence with its timestamp range."""
    text: str
    start: float
    end: float


@dataclass
class ProcessedTranscript:
    """Processed transcript with sentences and metadata."""
    video_id: str
    sentences: list[Sentence]
    full_text: str
    language: str
    word_count: int
    sentence_count: int
    quality_score: float
    preprocessing_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "video_id": self.video_id,
            "sentences": [asdict(s) for s in self.sentences],
            "full_text": self.full_text,
            "language": self.language,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "quality_score": self.quality_score,
            "preprocessing_version": self.preprocessing_version,
        }


class TranscriptPreprocessor:
    """Transform raw YouTube transcripts into clean, punctuated sentences."""

    # Filler words to remove (case-insensitive)
    # Conservative list: only clearly meaningless sounds/phrases to avoid false positives
    FILLERS_FR = [
        "euh", "euuh", "euhh",  # hesitation sounds
        "hum", "hmm", "mmh",    # thinking sounds
        "hein",                  # interjection
        "ben", "bah",           # informal fillers
        "tu vois", "tu sais",   # verbal tics
    ]
    FILLERS_EN = [
        "um", "uh", "uhh",      # hesitation sounds
        "hmm", "mm",            # thinking sounds
        "you know",             # verbal tic (phrase form)
        "i mean",               # verbal tic (phrase form)
        "kind of", "sort of",   # hedging (phrase form)
    ]

    # Non-speech markers to remove
    NON_SPEECH_MARKERS = [
        r"\[music\]",
        r"\[musique\]",
        r"\[applause\]",
        r"\[applaudissements\]",
        r"\[laughter\]",
        r"\[rires\]",
        r"\[inaudible\]",
        r"\[silence\]",
    ]

    # Abbreviations that don't end sentences
    ABBREVIATIONS = ["m.", "mr.", "mrs.", "ms.", "dr.", "prof.", "etc.", "ex.", "vs.", "ie.", "eg."]

    def __init__(self):
        """Initialize the preprocessor with punctuation model."""
        logger.info("Loading punctuation model...")
        self.punctuator = PunctuationModel()
        self.filler_pattern = self._compile_filler_pattern()
        self.non_speech_pattern = self._compile_non_speech_pattern()
        logger.info("Preprocessor initialized")

    def _compile_filler_pattern(self) -> re.Pattern:
        """Compile regex pattern for filler words."""
        all_fillers = self.FILLERS_FR + self.FILLERS_EN
        # Match whole words only
        pattern = r"\b(" + "|".join(re.escape(f) for f in all_fillers) + r")\b"
        return re.compile(pattern, re.IGNORECASE)

    def _compile_non_speech_pattern(self) -> re.Pattern:
        """Compile regex pattern for non-speech markers."""
        pattern = "|".join(self.NON_SPEECH_MARKERS)
        return re.compile(pattern, re.IGNORECASE)

    def merge_fragments(self, fragments: list[dict]) -> tuple[str, list[dict]]:
        """
        Merge transcript fragments into continuous text with position mapping.

        Args:
            fragments: List of {start, duration, text} dictionaries

        Returns:
            Tuple of (merged_text, position_map) where position_map maps
            character positions to timestamps
        """
        text_parts = []
        position_map = []
        current_pos = 0

        for frag in fragments:
            text = frag.get("text", "").strip()
            if not text:
                continue

            start_time = frag.get("start", 0.0)
            duration = frag.get("duration", 0.0)
            end_time = start_time + duration

            text_parts.append(text)
            position_map.append({
                "start_char": current_pos,
                "end_char": current_pos + len(text),
                "start_time": start_time,
                "end_time": end_time,
            })
            current_pos += len(text) + 1  # +1 for space separator

        merged_text = " ".join(text_parts)
        return merged_text, position_map

    def clean_text(self, text: str) -> tuple[str, dict[str, int]]:
        """
        Clean text by removing fillers and normalizing.

        Args:
            text: Raw merged text

        Returns:
            Tuple of (cleaned_text, stats) where stats contains cleaning metrics
        """
        original_length = len(text)
        stats = {"fillers_removed": 0, "non_speech_removed": 0, "repeated_words_removed": 0}

        # Count and remove non-speech markers
        non_speech_matches = self.non_speech_pattern.findall(text)
        stats["non_speech_removed"] = len(non_speech_matches)
        text = self.non_speech_pattern.sub("", text)

        # Count and remove filler words
        filler_matches = self.filler_pattern.findall(text)
        stats["fillers_removed"] = len(filler_matches)
        text = self.filler_pattern.sub("", text)

        # Remove repeated words (e.g., "le le problème" -> "le problème")
        repeated_pattern = r"\b(\w+)(\s+\1\b)+"
        repeated_matches = re.findall(repeated_pattern, text, re.IGNORECASE)
        stats["repeated_words_removed"] = len(repeated_matches)
        text = re.sub(repeated_pattern, r"\1", text, flags=re.IGNORECASE)

        # Remove false starts (word followed by dash and repeated)
        # e.g., "je vais— je vais parler" -> "je vais parler"
        text = re.sub(r"\b(\w+)\s*[—–-]\s*\1\b", r"\1", text, flags=re.IGNORECASE)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text, stats

    def add_punctuation(self, text: str) -> str:
        """
        Add punctuation to unpunctuated text using ML model.

        Args:
            text: Cleaned text without punctuation

        Returns:
            Text with punctuation and capitalization restored
        """
        if not text:
            return text

        # The model works best with reasonable chunk sizes
        # Split into chunks if text is very long
        max_chunk_size = 5000
        if len(text) <= max_chunk_size:
            return self.punctuator.restore_punctuation(text)

        # Process in chunks
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_len = len(word) + 1
            if current_length + word_len > max_chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_len
            else:
                current_chunk.append(word)
                current_length += word_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Punctuate each chunk
        punctuated_chunks = [self.punctuator.restore_punctuation(chunk) for chunk in chunks]
        return " ".join(punctuated_chunks)

    def apply_french_typography(self, text: str) -> str:
        """
        Apply French typography rules.

        In French, there's a space before : ; ! ?
        """
        # Add non-breaking space before French punctuation
        # Only if there isn't already a space
        text = re.sub(r"(\S)([;:!?])", r"\1 \2", text)
        # Fix double spaces
        text = re.sub(r"  +", " ", text)
        return text

    def segment_sentences(self, text: str, position_map: list[dict]) -> list[Sentence]:
        """
        Split text into sentences and map to timestamps.

        Args:
            text: Punctuated text
            position_map: Character position to timestamp mapping

        Returns:
            List of Sentence objects with timestamps
        """
        if not text or not position_map:
            return []

        sentences = []

        # Pattern to split on sentence boundaries
        # Handles . ! ? and combinations like ?! or ...
        # Avoids splitting on abbreviations and decimals
        sentence_boundaries = []
        current_pos = 0

        # Find sentence boundaries
        i = 0
        while i < len(text):
            char = text[i]

            # Check for sentence-ending punctuation
            if char in ".!?":
                # Check if it's not an abbreviation
                is_abbreviation = False
                word_start = text.rfind(" ", 0, i) + 1
                word_with_punct = text[word_start:i + 1].lower()

                if word_with_punct in self.ABBREVIATIONS:
                    is_abbreviation = True

                # Check if it's a decimal number
                if char == "." and i > 0 and i < len(text) - 1:
                    if text[i - 1].isdigit() and text[i + 1].isdigit():
                        i += 1
                        continue

                # Skip if it's part of ellipsis (handled below)
                if char == "." and i + 2 < len(text) and text[i:i + 3] == "...":
                    # Include all dots in the sentence
                    i += 3
                    sentence_boundaries.append(i)
                    continue

                if not is_abbreviation:
                    # Handle multiple punctuation (e.g., "?!" or "!!")
                    while i + 1 < len(text) and text[i + 1] in ".!?":
                        i += 1
                    sentence_boundaries.append(i + 1)

            i += 1

        # If no boundaries found, treat whole text as one sentence
        if not sentence_boundaries:
            sentence_boundaries = [len(text)]

        # Extract sentences
        start_idx = 0
        for end_idx in sentence_boundaries:
            sentence_text = text[start_idx:end_idx].strip()
            if sentence_text:
                # Find timestamps for this sentence
                start_time = self._find_timestamp(start_idx, position_map, "start")
                end_time = self._find_timestamp(end_idx - 1, position_map, "end")

                sentences.append(Sentence(
                    text=sentence_text,
                    start=start_time,
                    end=end_time,
                ))
            start_idx = end_idx

        return sentences

    def _find_timestamp(self, char_pos: int, position_map: list[dict], which: str) -> float:
        """
        Find timestamp for a character position.

        Args:
            char_pos: Character position in the text
            position_map: Position to timestamp mapping
            which: "start" or "end"

        Returns:
            Timestamp in seconds
        """
        if not position_map:
            return 0.0

        # Find the fragment containing this position
        for mapping in position_map:
            if mapping["start_char"] <= char_pos <= mapping["end_char"]:
                return mapping["start_time"] if which == "start" else mapping["end_time"]

        # If position is beyond mapped range, return last known timestamp
        if char_pos > position_map[-1]["end_char"]:
            return position_map[-1]["end_time"]

        # If position is before mapped range, return first timestamp
        return position_map[0]["start_time"]

    def detect_language(self, text: str) -> str:
        """
        Detect primary language of the text.

        Args:
            text: Text to analyze

        Returns:
            ISO language code (e.g., 'fr', 'en')
        """
        if not text or len(text) < 20:
            return "unknown"

        try:
            # Sample multiple sections for reliability
            sample_size = 500
            text_length = len(text)
            step = max(1, text_length // 5)

            samples = []
            for i in range(0, text_length, step):
                sample = text[i:i + sample_size]
                if len(sample) >= 50:
                    samples.append(sample)

            if not samples:
                return detect(text)

            # Detect language in each sample
            languages = []
            for sample in samples[:5]:  # Max 5 samples
                try:
                    lang = detect(sample)
                    languages.append(lang)
                except LangDetectException:
                    continue

            if not languages:
                return "unknown"

            # Return most common language
            return max(set(languages), key=languages.count)

        except LangDetectException:
            return "unknown"

    def calculate_quality_score(
        self,
        original_text: str,
        cleaned_text: str,
        sentences: list[Sentence],
        cleaning_stats: dict[str, int],
    ) -> float:
        """
        Calculate quality score for the transcript (0-100).

        Formula: 100 - (filler_ratio * 50) - (short_sentence_ratio * 30) - (repetition_ratio * 20)
        """
        if not original_text:
            return 0.0

        original_word_count = len(original_text.split())
        cleaned_word_count = len(cleaned_text.split())

        # Filler ratio (words removed / total words)
        words_removed = original_word_count - cleaned_word_count
        filler_ratio = words_removed / original_word_count if original_word_count > 0 else 0

        # Short sentence ratio (sentences with < 5 words)
        if sentences:
            short_sentences = sum(1 for s in sentences if len(s.text.split()) < 5)
            short_sentence_ratio = short_sentences / len(sentences)
        else:
            short_sentence_ratio = 0

        # Repetition ratio from cleaning stats
        total_issues = (
            cleaning_stats.get("repeated_words_removed", 0) +
            cleaning_stats.get("non_speech_removed", 0)
        )
        repetition_ratio = total_issues / original_word_count if original_word_count > 0 else 0

        # Calculate score
        score = 100 - (filler_ratio * 50) - (short_sentence_ratio * 30) - (repetition_ratio * 20)

        # Clamp to 0-100
        return max(0.0, min(100.0, round(score, 2)))

    def process(self, raw_transcript: dict[str, Any]) -> ProcessedTranscript:
        """
        Full preprocessing pipeline.

        Args:
            raw_transcript: Raw transcript dictionary with video_id and transcript array

        Returns:
            ProcessedTranscript object
        """
        video_id = raw_transcript.get("video_id", "unknown")
        fragments = raw_transcript.get("transcript", [])
        source_language = raw_transcript.get("language", "unknown")

        logger.info(f"Processing transcript for video {video_id} ({len(fragments)} fragments)")

        # Step 1: Merge fragments
        merged_text, position_map = self.merge_fragments(fragments)
        logger.debug(f"Merged {len(fragments)} fragments into {len(merged_text)} characters")

        # Step 2: Clean text
        cleaned_text, cleaning_stats = self.clean_text(merged_text)
        logger.debug(f"Cleaning stats: {cleaning_stats}")

        # Step 3: Add punctuation
        punctuated_text = self.add_punctuation(cleaned_text)

        # Step 4: Apply language-specific formatting
        detected_lang = self.detect_language(punctuated_text)
        language = detected_lang if detected_lang != "unknown" else source_language

        if language == "fr":
            punctuated_text = self.apply_french_typography(punctuated_text)

        # Step 5: Segment sentences
        sentences = self.segment_sentences(punctuated_text, position_map)
        logger.info(f"Segmented into {len(sentences)} sentences")

        # Step 6: Calculate quality score
        quality_score = self.calculate_quality_score(
            merged_text, cleaned_text, sentences, cleaning_stats
        )
        logger.info(f"Quality score: {quality_score}")

        return ProcessedTranscript(
            video_id=video_id,
            sentences=sentences,
            full_text=punctuated_text,
            language=language,
            word_count=len(punctuated_text.split()),
            sentence_count=len(sentences),
            quality_score=quality_score,
        )

    def process_file(self, input_path: Path, output_path: Path) -> ProcessedTranscript:
        """
        Process a single transcript file.

        Args:
            input_path: Path to raw transcript JSON
            output_path: Path to save processed transcript

        Returns:
            ProcessedTranscript object
        """
        logger.info(f"Processing file: {input_path}")

        # Load raw transcript
        raw_transcript = json.loads(input_path.read_text(encoding="utf-8"))

        # Process
        result = self.process(raw_transcript)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Saved processed transcript to: {output_path}")

        return result


def main():
    """CLI entry point for the transcript preprocessor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess YouTube transcripts for RAG pipeline"
    )
    parser.add_argument(
        "--video-id",
        help="Process a single video by ID",
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw_transcripts",
        help="Input directory containing raw transcripts",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_transcripts",
        help="Output directory for processed transcripts",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all raw transcripts in input directory",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = TranscriptPreprocessor()

    if args.video_id:
        # Process single video
        input_path = input_dir / f"{args.video_id}.json"
        output_path = output_dir / f"{args.video_id}.json"

        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return

        preprocessor.process_file(input_path, output_path)

    elif args.batch:
        # Process all transcripts
        input_files = list(input_dir.glob("*.json"))
        # Exclude manifest.json
        input_files = [f for f in input_files if f.name != "manifest.json"]

        if not input_files:
            logger.warning(f"No transcript files found in {input_dir}")
            return

        logger.info(f"Processing {len(input_files)} transcripts")

        success_count = 0
        error_count = 0

        for input_path in input_files:
            video_id = input_path.stem
            output_path = output_dir / f"{video_id}.json"

            try:
                preprocessor.process_file(input_path, output_path)
                success_count += 1
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                error_count += 1

        logger.info(f"Processing complete. Success: {success_count}, Errors: {error_count}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
