import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def split_into_chunks(text: str, chunk_size: int = 200) -> list[str]:
    """découpe le texte en chunks d'environ chunk_size mots, en respectant les frontières de phrases"""
    sentences = text.replace('\n', ' ').split('. ')
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for s in sentences:
        words = s.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append('. '.join(current))
            current = [s]
            current_len = len(words)
        else:
            current.append(s)
            current_len += len(words)

    if current:
        chunks.append('. '.join(current))

    return chunks if chunks else [text]


def embed_mean_chunks(
    text: str,
    model: SentenceTransformer,
    chunk_size: int = 200,
) -> np.ndarray:
    """encode chaque chunk et retourne la moyenne des embeddings"""
    chunks = split_into_chunks(text, chunk_size)
    embeddings = model.encode(chunks, show_progress_bar=False)
    return embeddings.mean(axis=0)


def score_max_chunks(
    cv_text: str,
    job_text: str,
    model: SentenceTransformer,
    chunk_size: int = 200,
) -> float:
    """retourne le score cosinus maximum entre les chunks du CV et l'offre entière"""
    cv_chunks = split_into_chunks(cv_text, chunk_size)
    cv_embeddings = model.encode(cv_chunks, show_progress_bar=False)
    job_embedding = model.encode([job_text], show_progress_bar=False)

    scores = cosine_similarity(cv_embeddings, job_embedding).flatten()
    return float(scores.max())
