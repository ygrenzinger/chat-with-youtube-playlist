#!/usr/bin/env python3
"""Extract plain text from YouTube JSON3 transcript, removing timing info."""

import json
from pathlib import Path


def extract_text_from_json3(input_path: Path) -> str:
    """Extract all text from a JSON3 transcript file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text_parts = []
    for event in data.get("events", []):
        for seg in event.get("segs", []):
            if "utf8" in seg:
                text_parts.append(seg["utf8"])

    # Join all parts and normalize to single line
    full_text = "".join(text_parts)
    # Replace newlines with spaces and collapse multiple spaces
    single_line = " ".join(full_text.split())
    return single_line


def main():
    input_file = Path("data/raw_transcripts/rxTTlMorMsA.en-orig.json3")
    output_file = input_file.with_suffix(".txt")

    text = extract_text_from_json3(input_file)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Extracted text saved to: {output_file}")
    print(f"Total characters: {len(text)}")


if __name__ == "__main__":
    main()
