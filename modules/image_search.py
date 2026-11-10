"""
PIXORA AI — Natural Language Image Search & Vector Retrieval
Powered by ChromaDB persistent vector database with vector similarity ranking.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

from modules.database import get_all_images, get_image
from modules.vector_store import search_vector_database, sync_all_existing_images_to_vector_store


def search_images(
    query: str,
    filter_type: str = "all",
    format_filter: Optional[str] = None,
    provider: Optional[Any] = None,
    db_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Search image library using natural language and vector similarity ranking.
    
    Workflow:
    1. Synchronize vector store to ensure all images are embedded.
    2. Query ChromaDB persistent collection with semantic query vector.
    3. Rank images by vector cosine similarity score.
    4. Apply metadata filters (favorites, format).
    5. Fallback to keyword matching if vector store has no entries.
    
    Returns:
        (results: List[Dict[str, Any]], active_search_mode: str)
    """
    clean_query = (query or "").strip()
    all_images = get_all_images(filter_type=filter_type, format_filter=format_filter, db_path=db_path)

    if not clean_query:
        return all_images, "All Library Images"

    # Map image_id -> image record for fast O(1) lookup
    image_lookup = {img["id"]: img for img in all_images}

    # 1. Primary: ChromaDB Persistent Vector Search
    vector_matches = search_vector_database(query_text=clean_query, top_k=20)
    
    if vector_matches:
        ranked_results: List[Dict[str, Any]] = []
        for img_id, sim_score in vector_matches:
            if img_id in image_lookup:
                record = dict(image_lookup[img_id])
                record["similarity_score"] = sim_score
                ranked_results.append(record)

        if ranked_results:
            mode_label = f"ChromaDB Vector Semantic Search (Ranked by Cosine Similarity)"
            return ranked_results, mode_label

    # 2. Secondary: Keyword & Tokenized Scoring Fallback
    terms = [re.sub(r"[^\w]", "", t).lower() for t in clean_query.split() if len(t) > 2]
    stop_words = {"show", "find", "pictures", "picture", "images", "image", "with", "from", "that", "this", "look", "like"}
    terms = [t for t in terms if t not in stop_words] or [clean_query.lower()]

    scored_results = []
    for img in all_images:
        score = 0.0
        caption = (img.get("caption") or "").lower()
        filename = (img.get("original_filename") or "").lower()

        for term in terms:
            if term in caption:
                score += 0.5
            if term in filename:
                score += 0.3

        if score > 0:
            rec = dict(img)
            rec["similarity_score"] = round(min(1.0, score), 3)
            scored_results.append((score, rec))

    scored_results.sort(key=lambda x: x[0], reverse=True)
    results = [item[1] for item in scored_results]

    mode_label = "Smart Caption & Keyword Match (Fallback)"
    return results, mode_label
