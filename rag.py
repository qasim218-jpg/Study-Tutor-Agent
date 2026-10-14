import math
import re
from collections import Counter


# Lightweight retrieval settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_INDEX_CHARS = 600_000
MAX_CHUNKS = 600
TOP_K = 3
MAX_RETRIEVED_CHARS = 2400

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text):
    """Convert text to lowercase searchable tokens."""
    return TOKEN_PATTERN.findall((text or "").lower())


def split_into_chunks(text):
    """Create bounded, overlapping text chunks."""

    text = re.sub(r"\s+", " ", text or "").strip()
    text = text[:MAX_INDEX_CHARS]

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text) and len(chunks) < MAX_CHUNKS:
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + 100:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def build_index(text):
    """
    Build a lightweight BM25 index without NumPy, PyTorch,
    sentence-transformers, or external embedding downloads.
    """

    chunks = split_into_chunks(text)

    if not chunks:
        raise ValueError(
            "No readable PDF text was found to index."
        )

    tokenized_chunks = []
    document_frequencies = Counter()
    document_lengths = []

    for chunk in chunks:
        tokens = tokenize(chunk)
        tokenized_chunks.append(Counter(tokens))
        document_lengths.append(len(tokens))
        document_frequencies.update(set(tokens))

    return {
        "chunks": chunks,
        "token_counts": tokenized_chunks,
        "document_frequencies": document_frequencies,
        "document_lengths": document_lengths,
        "average_document_length": (
            sum(document_lengths) / len(document_lengths)
            if document_lengths
            else 0
        ),
    }


def retrieve_passages(question, index):
    """Retrieve the highest-scoring passages using BM25."""

    if not index or not index.get("chunks"):
        return ""

    query_terms = set(tokenize(question))

    if not query_terms:
        return ""

    chunks = index["chunks"]
    token_counts = index["token_counts"]
    document_frequencies = index["document_frequencies"]
    document_lengths = index["document_lengths"]

    total_documents = len(chunks)
    average_length = index["average_document_length"]

    if not total_documents or not average_length:
        return ""

    k1 = 1.5
    b = 0.75
    scored_chunks = []

    for i, term_counts in enumerate(token_counts):
        score = 0.0
        doc_length = document_lengths[i]

        for term in query_terms:
            frequency = term_counts.get(term, 0)

            if frequency == 0:
                continue

            df = document_frequencies.get(term, 0)

            inverse_document_frequency = math.log(
                1
                + (
                    total_documents - df + 0.5
                ) / (
                    df + 0.5
                )
            )

            denominator = frequency + k1 * (
                1 - b + b * doc_length / average_length
            )

            score += inverse_document_frequency * (
                frequency * (k1 + 1)
            ) / denominator

        if score > 0:
            scored_chunks.append((score, i))

    if not scored_chunks:
        return ""

    scored_chunks.sort(reverse=True)

    selected = []
    total_chars = 0

    for _, idx in scored_chunks[:TOP_K]:
        chunk = chunks[idx]
        remaining = MAX_RETRIEVED_CHARS - total_chars

        if remaining <= 0:
            break

        excerpt = chunk[:remaining].strip()

        if excerpt:
            selected.append(excerpt)
            total_chars += len(excerpt)

    return "\n\n---\n\n".join(selected)
