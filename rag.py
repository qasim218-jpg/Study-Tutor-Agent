import re

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
MAX_CHUNKS = 3000
TOP_K = 3
MAX_RETRIEVED_CHARS = 2400


@st.cache_resource(show_spinner="Loading local embedding model...")
def load_embedding_model():
    """Load the embedding model once per running app process."""
    return SentenceTransformer(MODEL_NAME)


def split_into_chunks(text):
    """Split extracted PDF text into overlapping passages."""

    text = re.sub(r"\s+", " ", text or "").strip()

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
    """Create an in-memory semantic search index for a PDF."""

    chunks = split_into_chunks(text)

    if not chunks:
        raise ValueError(
            "No readable PDF text was found to index."
        )

    model = load_embedding_model()
    embeddings = model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return {
        "chunks": chunks,
        "embeddings": np.asarray(
            embeddings,
            dtype=np.float32,
        ),
    }


def retrieve_passages(question, index):
    """Return a small, relevant set of PDF passages."""

    if not index or not index.get("chunks"):
        return ""

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    scores = index["embeddings"] @ query_embedding

    # Examine the best candidates, then preserve their document order.
    candidate_count = min(TOP_K, len(scores))
    best_indices = np.argsort(scores)[-candidate_count:]
    best_indices = sorted(
        best_indices,
        key=lambda idx: scores[idx],
        reverse=True,
    )

    selected = []
    total_chars = 0

    for idx in best_indices:
        chunk = index["chunks"][int(idx)]
        remaining = MAX_RETRIEVED_CHARS - total_chars

        if remaining <= 0:
            break

        excerpt = chunk[:remaining].strip()

        if excerpt:
            selected.append(excerpt)
            total_chars += len(excerpt)

    return "\n\n---\n\n".join(selected)
