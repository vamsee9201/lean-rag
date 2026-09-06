#!/usr/bin/env python3
"""Build and query a reproducible SQLite FTS5/BM25 index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAGES = ROOT / "data" / "processed" / "pilot30" / "pages.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "indexes" / "pilot30-bm25.sqlite3"
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "did", "do", "does", "for", "from", "how", "if", "in", "is", "it", "of", "on",
    "or", "than", "that", "the", "then", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "why", "with", "according", "document", "page",
    "report", "specific", "text", "exact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-words", type=int, default=350)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def chunks_for_page(page: dict, chunk_words: int, overlap_words: int):
    words = page["text"].split()
    if not words:
        return
    step = chunk_words - overlap_words
    if step < 1:
        raise ValueError("--overlap-words must be smaller than --chunk-words")
    for index, start in enumerate(range(0, len(words), step)):
        chunk = words[start : start + chunk_words]
        if not chunk:
            continue
        yield {
            "chunk_id": f"{page['document_id']}:p{page['page']}:c{index}",
            "document_id": page["document_id"],
            "title": page["title"],
            "collection": page["collection"],
            "page_start": page["page"],
            "page_end": page["page"],
            "text": " ".join(chunk),
        }
        if start + chunk_words >= len(words):
            break


def create_index(pages_path: Path, output_path: Path, chunk_words: int, overlap_words: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    try:
        db.executescript(
            """
            CREATE TABLE chunks(
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                collection TEXT NOT NULL,
                page_start INTEGER NOT NULL,
                page_end INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                title UNINDEXED,
                text,
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        chunk_count = 0
        document_ids: set[str] = set()
        with pages_path.open(encoding="utf-8") as stream:
            for line in stream:
                page = json.loads(line)
                for chunk in chunks_for_page(page, chunk_words, overlap_words):
                    db.execute(
                        "INSERT INTO chunks VALUES(?,?,?,?,?,?,?)",
                        tuple(chunk[key] for key in (
                            "chunk_id", "document_id", "title", "collection",
                            "page_start", "page_end", "text"
                        )),
                    )
                    db.execute(
                        "INSERT INTO chunks_fts VALUES(?,?,?)",
                        (chunk["chunk_id"], chunk["title"], chunk["text"]),
                    )
                    document_ids.add(chunk["document_id"])
                    chunk_count += 1
        metadata = {
            "pages_path": str(pages_path),
            "chunk_words": str(chunk_words),
            "overlap_words": str(overlap_words),
            "documents": str(len(document_ids)),
            "chunks": str(chunk_count),
        }
        db.executemany("INSERT INTO metadata VALUES(?,?)", metadata.items())
        db.commit()
    finally:
        db.close()
    temporary.replace(output_path)
    print(json.dumps(metadata, indent=2))


def fts_query(text: str) -> str:
    terms = re.findall(r"[\w]+", text, flags=re.UNICODE)
    if not terms:
        raise ValueError("Query contains no searchable terms")
    content_terms = [term for term in terms if len(term) > 2 and term.casefold() not in STOPWORDS]
    terms = content_terms or terms
    return " OR ".join(f'"{term}"' for term in terms)


def search(index_path: Path, query: str, top_k: int) -> list[dict]:
    db = sqlite3.connect(index_path)
    db.row_factory = sqlite3.Row
    try:
        rows = db.execute(
            """
            SELECT c.*, bm25(chunks_fts, 0.0, 0.0, 1.0) AS score
            FROM chunks_fts
            JOIN chunks c USING(chunk_id)
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query(query), top_k),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        db.close()


def main() -> None:
    args = parse_args()
    if args.force or not args.output.exists():
        create_index(args.pages, args.output, args.chunk_words, args.overlap_words)
    if args.query:
        print(json.dumps(search(args.output, args.query, args.top_k), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
