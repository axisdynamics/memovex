"""
memovex — Data Ingestion Script.

Ingests structured data (JSONL / CSV) into MemoryBank.
Supports batch embedding when sentence-transformers is available.

Usage:
    python scripts/ingest_data.py --file data.jsonl [--type episodic]
    python scripts/ingest_data.py --file data.csv --text-col content

Format for JSONL:
    {"text": "...", "entities": ["A", "B"], "tags": ["x"], "confidence": 0.8}

Format for CSV:
    text,entities,tags,confidence
    "...",A|B,x,0.8
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from memovex.core.memory_bank import MemoVexOrchestrator
from memovex.core.types import MemoryType

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest data into memovex")
    p.add_argument("--file", required=True, help="Input file (.jsonl or .csv)")
    p.add_argument("--type", default="episodic",
                   choices=["episodic", "semantic", "wisdom", "procedural"],
                   help="Memory type for all records")
    p.add_argument("--text-col", default="text",
                   help="Column name for text in CSV mode")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for embedding generation")
    p.add_argument("--no-embeddings", action="store_true",
                   help="Disable embedding generation (faster, BoW only)")
    p.add_argument("--confidence", type=float, default=0.7,
                   help="Default confidence when not in record")
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping line %d: %s", i + 1, e)
    return records


def load_csv(path: Path, text_col: str) -> List[Dict]:
    records = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_col, "").strip()
            if not text:
                continue
            entities_raw = row.get("entities", "")
            tags_raw = row.get("tags", "")
            records.append({
                "text": text,
                "entities": [e.strip() for e in entities_raw.split("|") if e.strip()],
                "tags": [t.strip() for t in tags_raw.split("|") if t.strip()],
                "confidence": float(row.get("confidence", 0.7)),
            })
    return records


def ingest(records: List[Dict], bank: MemoVexOrchestrator,
           memory_type: MemoryType, default_confidence: float) -> int:
    stored = 0
    t0 = time.time()
    for i, rec in enumerate(records):
        text = rec.get("text", "").strip()
        if not text:
            continue
        try:
            bank.store(
                text=text,
                memory_type=memory_type,
                entities=set(rec.get("entities", [])),
                tags=set(rec.get("tags", [])),
                confidence=rec.get("confidence", default_confidence),
                salience=rec.get("salience", 0.5),
            )
            stored += 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                logger.info("  %d/%d stored (%.1fs)", i + 1, len(records), elapsed)
        except Exception as e:
            logger.warning("Failed to store record %d: %s", i, e)

    return stored


def main() -> None:
    args = parse_args()
    path = Path(args.file)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    logger.info("Loading records from %s", path)
    if path.suffix == ".jsonl":
        records = load_jsonl(path)
    elif path.suffix == ".csv":
        records = load_csv(path, args.text_col)
    else:
        logger.error("Unsupported format: %s (use .jsonl or .csv)", path.suffix)
        sys.exit(1)

    logger.info("Loaded %d records", len(records))

    memory_type = MemoryType(args.type)

    bank = MemoVexOrchestrator(embeddings_enabled=not args.no_embeddings)
    bank.initialize()

    logger.info("Ingesting into memovex (type=%s, embeddings=%s)...",
                args.type, not args.no_embeddings)
    t0 = time.time()
    stored = ingest(records, bank, memory_type, args.confidence)
    elapsed = time.time() - t0

    logger.info("Done. Stored %d/%d records in %.2fs (%.0f rec/s)",
                stored, len(records), elapsed,
                stored / max(elapsed, 0.001))
    logger.info("Memory stats: %s", bank.stats())

    bank.shutdown()


if __name__ == "__main__":
    main()
