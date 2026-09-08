"""Dependency-light validation helpers for dense-retriever supervision."""

from __future__ import annotations

from collections import defaultdict


def compact(text: str) -> str:
    return " ".join(text.split()).casefold()


def resolve_positives(question: dict, by_page: dict[tuple[str, int], list[dict]]) -> list[str]:
    positives = []
    for document, page, quotation in zip(
        question["gold_documents"], question["gold_pages"], question["gold_passages"]
    ):
        quote_terms = set(compact(quotation).split())
        candidates = by_page.get((document, page), [])
        exact = [row for row in candidates if compact(quotation) in compact(row["text"])]
        if exact:
            positives.append(exact[0]["chunk_id"])
            continue
        scored = sorted(
            candidates,
            key=lambda row: (
                -(len(quote_terms & set(compact(row["text"]).split())) / max(1, len(quote_terms))),
                row["chunk_id"],
            ),
        )
        if not scored:
            raise ValueError(f"No chunks on gold page for {question['question_id']}")
        overlap = len(quote_terms & set(compact(scored[0]["text"]).split())) / max(1, len(quote_terms))
        if overlap < 0.80:
            raise ValueError(f"Cannot resolve gold chunk for {question['question_id']}: {overlap:.3f}")
        positives.append(scored[0]["chunk_id"])
    return list(dict.fromkeys(positives))


def valid_negative(candidate: dict, question: dict, positives: set[str]) -> bool:
    if candidate["chunk_id"] in positives:
        return False
    text = compact(candidate["text"])
    if any(compact(quote) in text for quote in question["gold_passages"]):
        return False
    answer = compact(question["reference_answer"])
    if len(answer) >= 4 and answer in text:
        return False
    positive_terms = [set(compact(quote).split()) for quote in question["gold_passages"]]
    terms = set(text.split())
    if any(len(terms & gold) / max(1, len(terms | gold)) >= 0.90 for gold in positive_terms):
        return False
    return True
