"""
PIXORA AI — Natural Language Image Search
Hybrid search engine combining vector embeddings and SQLite full-text search.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

from modules.database import get_all_embeddings, get_all_images, get_connection
from modules.providers import BaseAIProvider, get_provider
from modules.utilities import cosine_similarity


def search_images(
    query: str,
    filter_type: str = "all",
    format_filter: Optional[str] = None,
    provider: Optional[BaseAIProvider] = None,
    db_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Search image library using natural language.
    
    Returns:
        (results: List[Dict[str, Any]], active_search_mode: str)
    """
    clean_query = (query or "").strip()
    if not clean_query:
        # Return standard filtered library
        return get_all_images(filter_type=filter_type, format_filter=format_filter, db_path=db_path), "All Library Images"

    active_provider = provider or get_provider()

    # 1. Attempt Vector Semantic Search if provider supports embeddings and keys are present
    if active_provider.is_configured():
        try:
            query_embedding = active_provider.generate_embedding(clean_query)
            if query_embedding:
                stored_embeddings = get_all_embeddings(db_path=db_path)
                if stored_embeddings:
                    # Calculate cosine similarity
                    scored = []
                    for item in stored_embeddings:
                        sim = cosine_similarity(query_embedding, item["embedding"])
                        scored.append((item["image_id"], sim))
                    
                    scored.sort(key=lambda x: x[1], reverse=True)
                    top_matches = [img_id for img_id, score in scored if score > 0.35]

                    if top_matches:
                        all_imgs = {img["id"]: img for img in get_all_images(filter_type=filter_type, format_filter=format_filter, db_path=db_path)}
                        matched_imgs = [all_imgs[mid] for mid in top_matches if mid in all_imgs]
                        mode_label = f"Semantic Embedding Search ({active_provider.provider_name.capitalize()})"
                        return matched_imgs, mode_label
        except Exception:
            # Gracefully proceed to fallback
            pass

    # 2. SQLite Full-Text / Keyword Search Fallback
    # Extract search terms and clean punctuation
    terms = [re.sub(r"[^\w]", "", t).lower() for t in clean_query.split() if len(t) > 2]
    # Filter out common filler words
    stop_words = {"show", "find", "pictures", "picture", "images", "image", "with", "from", "that", "this", "look", "like"}
    terms = [t for t in terms if t not in stop_words]

    if not terms:
        terms = [clean_query.lower()]

    all_images = get_all_images(filter_type=filter_type, format_filter=format_filter, db_path=db_path)
    scored_results = []

    for img in all_images:
        score = 0
        caption = (img.get("caption") or "").lower()
        filename = (img.get("original_filename") or "").lower()

        # Score matching terms
        for term in terms:
            if term in caption:
                score += 3
            if term in filename:
                score += 2

        if score > 0:
            scored_results.append((score, img))

    # Sort descending by match score
    scored_results.sort(key=lambda x: x[0], reverse=True)
    results = [item[1] for item in scored_results]

    mode_label = "Smart Caption & Keyword Search (SQLite Fallback)"
    return results, mode_label
