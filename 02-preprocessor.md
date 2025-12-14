# Component: Transcript Preprocessor

## Purpose

Transform raw YouTube transcript fragments into clean, punctuated, sentence-structured text while preserving timestamp alignment.

## Input

Raw transcript JSON from the extractor:

```json
{
  "transcript": [
    {"start": 0.0, "duration": 2.4, "text": "bonjour à tous"},
    {"start": 2.4, "duration": 3.1, "text": "aujourd'hui on va parler de"},
    {"start": 5.5, "duration": 2.8, "text": "systèmes rag et comment"},
    {"start": 8.3, "duration": 2.2, "text": "les construire efficacement"}
  ]
}
```

## Output

Processed transcript with sentences and preserved timestamps:

```json
{
  "video_id": "abc123xyz",
  "sentences": [
    {
      "text": "Bonjour à tous.",
      "start": 0.0,
      "end": 2.4
    },
    {
      "text": "Aujourd'hui, on va parler de systèmes RAG et comment les construire efficacement.",
      "start": 2.4,
      "end": 10.5
    }
  ],
  "full_text": "Bonjour à tous. Aujourd'hui, on va parler de systèmes RAG...",
  "language": "fr",
  "word_count": 245,
  "sentence_count": 12,
  "preprocessing_version": "1.0"
}
```

## Functional Requirements

### 1. Fragment Merging

- Concatenate consecutive fragments into continuous text
- Preserve original timestamps (map sentences back to time ranges)
- Handle overlapping timestamps gracefully

**Algorithm:**
```
fragments: [(0.0, 2.4, "hello"), (2.4, 3.1, "everyone today"), (5.5, 2.8, "we will")]
           ↓
merged:    "hello everyone today we will"
           ↓
with timestamps: each character position maps to original fragment time
```

### 2. Punctuation Restoration

Auto-generated captions lack punctuation. Restore:
- Periods (.)
- Commas (,)
- Question marks (?)
- Exclamation points (!)
- Capitalization at sentence starts

**Recommended approaches:**

| Approach | Model/Tool | Pros | Cons |
|----------|-----------|------|------|
| Deep Punctuation | `deepmultilingualpunctuation` | Fast, multilingual | Less accurate |
| NNSplit | `nnsplit` | Good sentence splitting | No comma restoration |
| LLM-based | Claude/GPT-4 | Most accurate | Slow, costly |
| Hybrid | NNSplit + rules | Good balance | Requires tuning |

**French-specific considerations:**
- Handle French quotation marks (« »)
- Proper spacing before `:`, `;`, `!`, `?` (French typography)
- Contractions: "aujourd'hui", "c'est", "qu'est-ce"

### 3. Text Cleaning

**Remove or normalize:**

| Pattern | Action | Example |
|---------|--------|---------|
| Filler words | Remove | "euh", "hum", "ben", "you know", "like" |
| False starts | Remove | "je vais— je vais parler" → "je vais parler" |
| Repeated words | Dedupe | "le le problème" → "le problème" |
| Extra whitespace | Normalize | Multiple spaces → single space |
| Special chars | Clean | "[Music]", "[Applause]" → remove or tag |

**Keep meaningful discourse markers:**
- "donc" (so/therefore)
- "alors" (then/so)
- "par exemple" (for example)
- "en fait" (actually)
- "c'est-à-dire" (that is to say)

### 4. Sentence Segmentation

Split text into sentences using restored punctuation:

```python
def segment_sentences(text: str, fragments: list) -> list:
    """
    Split text into sentences and map back to timestamps.
    
    Returns:
        List of {text, start, end} dictionaries
    """
    # Use punctuation boundaries
    # Handle abbreviations (M., Dr., etc.)
    # Handle decimal numbers (3.14 is not a sentence boundary)
    # Handle URLs mentioned verbally
```

**Timestamp alignment strategy:**

When merging fragments and adding punctuation, timestamps become approximate:
- Sentence start = start time of first fragment containing sentence start
- Sentence end = end time of last fragment containing sentence end
- For retrieval purposes, ±2 seconds accuracy is acceptable

### 5. Language Detection

- Detect primary language per video
- Handle code-switching (French talk with English technical terms)
- Tag language at video level, not sentence level

```python
from langdetect import detect

def detect_language(text: str) -> str:
    # Sample multiple sections for reliability
    samples = [text[i:i+500] for i in range(0, len(text), len(text)//5)]
    languages = [detect(s) for s in samples if len(s) > 100]
    return max(set(languages), key=languages.count)
```

### 6. Quality Metrics

Calculate and store for each transcript:

| Metric | Description | Use |
|--------|-------------|-----|
| `word_count` | Total words | Estimate processing time |
| `sentence_count` | Total sentences | Chunking planning |
| `avg_sentence_length` | Words per sentence | Quality indicator |
| `filler_ratio` | Fillers / total words | Transcript quality |
| `quality_score` | Composite 0-100 | Flag for review |

**Quality score formula:**
```python
quality_score = 100 - (filler_ratio * 50) - (short_sentence_ratio * 30) - (repetition_ratio * 20)
```

## Technical Constraints

- Must be deterministic (same input → same output)
- Process one video at a time (memory efficiency)
- Support batch processing for full channel reprocessing
- Preserve original data (store raw alongside processed)

## Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Single-word fragments | Merge aggressively |
| Very long speech (5+ min no pause) | Force sentence breaks at natural points |
| Technical jargon | Preserve as-is, don't "correct" |
| Speaker changes | Cannot detect reliably, ignore |
| Music/applause segments | Tag as `[non-speech]` or remove |
| Numbers and dates | Preserve formatting |

## Example Implementation Skeleton

```python
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class Sentence:
    text: str
    start: float
    end: float

@dataclass
class ProcessedTranscript:
    video_id: str
    sentences: List[Sentence]
    full_text: str
    language: str
    word_count: int
    sentence_count: int
    quality_score: float
    preprocessing_version: str = "1.0"

class TranscriptPreprocessor:
    def __init__(self, punctuation_model: str = "deepmultilingualpunctuation"):
        self.punctuator = self._load_punctuator(punctuation_model)
        self.filler_patterns = self._compile_filler_patterns()
    
    def _compile_filler_patterns(self) -> re.Pattern:
        fillers_fr = ["euh", "hum", "ben", "bah", "genre", "voilà", "quoi"]
        fillers_en = ["um", "uh", "like", "you know", "basically", "literally"]
        all_fillers = fillers_fr + fillers_en
        return re.compile(r'\b(' + '|'.join(all_fillers) + r')\b', re.IGNORECASE)
    
    def merge_fragments(self, fragments: list) -> tuple[str, list]:
        """Merge fragments into continuous text with position mapping."""
        text_parts = []
        position_map = []  # Maps character positions to timestamps
        
        current_pos = 0
        for frag in fragments:
            text = frag["text"].strip()
            if text:
                text_parts.append(text)
                position_map.append({
                    "start_char": current_pos,
                    "end_char": current_pos + len(text),
                    "start_time": frag["start"],
                    "end_time": frag["start"] + frag["duration"]
                })
                current_pos += len(text) + 1  # +1 for space
        
        return " ".join(text_parts), position_map
    
    def clean_text(self, text: str) -> str:
        """Remove fillers and normalize text."""
        # Remove fillers
        text = self.filler_patterns.sub("", text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove repeated words
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
        return text.strip()
    
    def add_punctuation(self, text: str) -> str:
        """Add punctuation using the configured model."""
        return self.punctuator.punctuate(text)
    
    def segment_sentences(self, text: str, position_map: list) -> List[Sentence]:
        """Split into sentences with timestamps."""
        # Simple regex-based splitting (enhance as needed)
        sentence_pattern = re.compile(r'[^.!?]*[.!?]')
        sentences = []
        
        for match in sentence_pattern.finditer(text):
            sent_text = match.group().strip()
            start_char = match.start()
            end_char = match.end()
            
            # Find corresponding timestamps
            start_time = self._find_timestamp(start_char, position_map, "start")
            end_time = self._find_timestamp(end_char, position_map, "end")
            
            sentences.append(Sentence(
                text=sent_text,
                start=start_time,
                end=end_time
            ))
        
        return sentences
    
    def _find_timestamp(self, char_pos: int, position_map: list, which: str) -> float:
        """Find timestamp for a character position."""
        for mapping in position_map:
            if mapping["start_char"] <= char_pos <= mapping["end_char"]:
                return mapping["start_time"] if which == "start" else mapping["end_time"]
        return position_map[-1]["end_time"]  # Fallback to end
    
    def calculate_quality_score(self, original: str, cleaned: str) -> float:
        """Calculate quality score 0-100."""
        filler_ratio = 1 - (len(cleaned) / len(original)) if original else 0
        return max(0, 100 - (filler_ratio * 100))
    
    def process(self, raw_transcript: dict) -> ProcessedTranscript:
        """Full preprocessing pipeline."""
        # 1. Merge fragments
        merged_text, position_map = self.merge_fragments(raw_transcript["transcript"])
        
        # 2. Clean text
        cleaned_text = self.clean_text(merged_text)
        
        # 3. Add punctuation
        punctuated_text = self.add_punctuation(cleaned_text)
        
        # 4. Segment sentences
        sentences = self.segment_sentences(punctuated_text, position_map)
        
        # 5. Calculate metrics
        quality_score = self.calculate_quality_score(merged_text, cleaned_text)
        
        return ProcessedTranscript(
            video_id=raw_transcript["video_id"],
            sentences=sentences,
            full_text=punctuated_text,
            language=raw_transcript.get("language", "unknown"),
            word_count=len(punctuated_text.split()),
            sentence_count=len(sentences),
            quality_score=quality_score
        )
```

## Dependencies

```python
# requirements.txt
deepmultilingualpunctuation>=1.0.0
langdetect>=1.0.9
regex>=2023.0.0
```

## Output Location

- Processed transcripts: `data/processed_transcripts/{video_id}.json`

## Success Criteria

- [ ] Punctuation is restored with >90% accuracy
- [ ] Sentences are properly segmented
- [ ] Timestamps are accurate within ±2 seconds
- [ ] Filler words are removed without losing meaning
- [ ] Quality scores correctly identify problematic transcripts
- [ ] Processing is deterministic and reproducible
