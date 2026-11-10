import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

from config import Config


def get_db_path(db_path: Optional[Path] = None) -> Path:
    """Return configured database path or override."""
    target_path = Path(db_path) if db_path else Config.DATABASE_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open SQLite connection with row factory and foreign keys enabled."""
    conn = sqlite3.connect(get_db_path(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables, indexes, and full-text search index."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Images table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                original_path TEXT NOT NULL,
                thumbnail_path TEXT NOT NULL,
                caption TEXT,
                upload_timestamp TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                file_size INTEGER NOT NULL,
                active_version_id TEXT,
                is_favorite INTEGER DEFAULT 0
            );
            """
        )

        # 2. Image versions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS image_versions (
                id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                version_path TEXT NOT NULL,
                thumbnail_path TEXT NOT NULL,
                prompt TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                parent_version_id TEXT,
                created_at TEXT NOT NULL,
                is_original INTEGER DEFAULT 0,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
            );
            """
        )

        # 3. Image reverts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS image_reverts (
                id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL,
                from_version_id TEXT NOT NULL,
                to_version_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
            );
            """
        )

        # 4. Image embeddings table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS image_embeddings (
                id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL,
                embedding_provider TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE
            );
            """
        )

        # Create performance indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_upload ON images(upload_timestamp DESC);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_versions_img ON image_versions(image_id, version_number);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reverts_img ON image_reverts(image_id);")

        # Create FTS5 virtual table for keyword search if supported
        try:
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS image_search_idx USING fts5(
                    image_id UNINDEXED,
                    caption,
                    original_filename,
                    prompts
                );
                """
            )
        except sqlite3.OperationalError:
            # Fallback if SQLite build lacks FTS5 extension
            pass

        conn.commit()


# --- Image CRUD Operations ---

def insert_image(
    image_id: str,
    original_filename: str,
    stored_filename: str,
    original_path: str,
    thumbnail_path: str,
    mime_type: str,
    width: int,
    height: int,
    file_size: int,
    caption: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Insert a new image record and update search index."""
    now_iso = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO images (
                id, original_filename, stored_filename, original_path,
                thumbnail_path, caption, upload_timestamp, mime_type,
                width, height, file_size, active_version_id, is_favorite
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)
            """,
            (
                image_id,
                original_filename,
                stored_filename,
                str(original_path),
                str(thumbnail_path),
                caption or "",
                now_iso,
                mime_type,
                width,
                height,
                file_size,
            ),
        )

        # Index in FTS5
        try:
            cursor.execute(
                """
                INSERT OR REPLACE INTO image_search_idx (image_id, caption, original_filename, prompts)
                VALUES (?, ?, ?, ?)
                """,
                (image_id, caption or "", original_filename, "Original Upload"),
            )
        except sqlite3.OperationalError:
            pass

        conn.commit()

    return get_image(image_id, db_path=db_path)  # type: ignore


def get_image(image_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve an image by ID with its active version details."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        if not row:
            return None
        data = dict(row)

        # Fetch active version record if present
        if data.get("active_version_id"):
            cursor.execute("SELECT * FROM image_versions WHERE id = ?", (data["active_version_id"],))
            ver_row = cursor.fetchone()
            data["active_version"] = dict(ver_row) if ver_row else None
        else:
            data["active_version"] = None

        return data


def update_image_caption(image_id: str, caption: str, db_path: Optional[Path] = None) -> None:
    """Update caption for an image and refresh search index."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE images SET caption = ? WHERE id = ?", (caption, image_id))
        try:
            cursor.execute(
                "UPDATE image_search_idx SET caption = ? WHERE image_id = ?",
                (caption, image_id),
            )
        except sqlite3.OperationalError:
            pass
        conn.commit()


def update_image_active_version(image_id: str, version_id: str, db_path: Optional[Path] = None) -> None:
    """Set the active version pointer for an image."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE images SET active_version_id = ? WHERE id = ?", (version_id, image_id))
        conn.commit()


def toggle_image_favorite(image_id: str, db_path: Optional[Path] = None) -> bool:
    """Toggle the favorite status of an image. Return the new status."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT is_favorite FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        if not row:
            return False
        new_val = 0 if row["is_favorite"] else 1
        cursor.execute("UPDATE images SET is_favorite = ? WHERE id = ?", (new_val, image_id))
        conn.commit()
        return bool(new_val)


def get_all_images(
    filter_type: str = "all",
    format_filter: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all images with version count and active thumbnail."""
    query = """
        SELECT 
            i.*,
            COUNT(v.id) as version_count,
            COALESCE(av.version_path, i.original_path) as current_display_path,
            COALESCE(av.thumbnail_path, i.thumbnail_path) as current_thumbnail_path,
            av.version_number as current_version_number
        FROM images i
        LEFT JOIN image_versions v ON i.id = v.image_id
        LEFT JOIN image_versions av ON i.active_version_id = av.id
    """
    conditions = []
    params = []

    if filter_type == "favorites":
        conditions.append("i.is_favorite = 1")
    elif filter_type == "recent_edits":
        conditions.append("i.active_version_id IS NOT NULL")

    if format_filter and format_filter != "all":
        conditions.append("LOWER(i.mime_type) LIKE ?")
        params.append(f"%{format_filter.lower()}%")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " GROUP BY i.id"

    if filter_type == "recent_edits":
        query += " ORDER BY COALESCE(av.created_at, i.upload_timestamp) DESC"
    else:
        query += " ORDER BY i.upload_timestamp DESC"

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def delete_image(image_id: str, db_path: Optional[Path] = None) -> List[Path]:
    """
    Delete image from database and return a list of all associated file paths 
    (original, thumbnails, and version files) to be cleaned up from disk.
    """
    files_to_delete: List[Path] = []
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Collect paths before deletion
        cursor.execute("SELECT original_path, thumbnail_path FROM images WHERE id = ?", (image_id,))
        img_row = cursor.fetchone()
        if img_row:
            if img_row["original_path"]:
                files_to_delete.append(Path(img_row["original_path"]))
            if img_row["thumbnail_path"]:
                files_to_delete.append(Path(img_row["thumbnail_path"]))

        cursor.execute("SELECT version_path, thumbnail_path FROM image_versions WHERE image_id = ?", (image_id,))
        for ver in cursor.fetchall():
            if ver["version_path"]:
                files_to_delete.append(Path(ver["version_path"]))
            if ver["thumbnail_path"]:
                files_to_delete.append(Path(ver["thumbnail_path"]))

        # Cascading delete removes versions, reverts, embeddings
        cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        try:
            cursor.execute("DELETE FROM image_search_idx WHERE image_id = ?", (image_id,))
        except sqlite3.OperationalError:
            pass

        # Delete vector embedding from ChromaDB persistent vector database
        try:
            from modules.vector_store import delete_image_vector
            delete_image_vector(image_id)
        except Exception:
            pass

        conn.commit()

    return files_to_delete


# --- Version History CRUD Operations ---

def insert_version(
    version_id: str,
    image_id: str,
    version_number: int,
    version_path: str,
    thumbnail_path: str,
    prompt: str,
    provider: str,
    model: str,
    parent_version_id: Optional[str] = None,
    is_original: bool = False,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Insert a new image version and set as active version."""
    now_iso = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO image_versions (
                id, image_id, version_number, version_path, thumbnail_path,
                prompt, provider, model, parent_version_id, created_at, is_original
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version_id,
                image_id,
                version_number,
                str(version_path),
                str(thumbnail_path),
                prompt,
                provider,
                model,
                parent_version_id,
                now_iso,
                1 if is_original else 0,
            ),
        )

        # Update active pointer on parent image
        cursor.execute("UPDATE images SET active_version_id = ? WHERE id = ?", (version_id, image_id))

        # Update search index prompts accumulator
        try:
            cursor.execute("SELECT prompts FROM image_search_idx WHERE image_id = ?", (image_id,))
            s_row = cursor.fetchone()
            if s_row:
                curr_prompts = s_row["prompts"] or ""
                updated_prompts = f"{curr_prompts} | {prompt}" if curr_prompts else prompt
                cursor.execute(
                    "UPDATE image_search_idx SET prompts = ? WHERE image_id = ?",
                    (updated_prompts, image_id),
                )
        except sqlite3.OperationalError:
            pass

        conn.commit()

    return get_version(version_id, db_path=db_path)  # type: ignore


def get_version(version_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve version record by ID."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM image_versions WHERE id = ?", (version_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_versions_for_image(image_id: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all versions of an image ordered by version_number ASC."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM image_versions WHERE image_id = ? ORDER BY version_number ASC",
            (image_id,),
        )
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def get_next_version_number(image_id: str, db_path: Optional[Path] = None) -> int:
    """Return the next sequential version number for an image."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(version_number) as max_v FROM image_versions WHERE image_id = ?",
            (image_id,),
        )
        row = cursor.fetchone()
        if row and row["max_v"] is not None:
            return int(row["max_v"]) + 1
        return 1


def record_revert(
    revert_id: str,
    image_id: str,
    from_version_id: str,
    to_version_id: str,
    db_path: Optional[Path] = None,
) -> None:
    """Record an audit trail entry for a revert event and update active pointer."""
    now_iso = datetime.utcnow().isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO image_reverts (id, image_id, from_version_id, to_version_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (revert_id, image_id, from_version_id, to_version_id, now_iso),
        )
        cursor.execute("UPDATE images SET active_version_id = ? WHERE id = ?", (to_version_id, image_id))
        conn.commit()


# --- Embeddings CRUD Operations ---

def save_image_embedding(
    embedding_id: str,
    image_id: str,
    provider: str,
    model: str,
    embedding: List[float],
    db_path: Optional[Path] = None,
) -> None:
    """Store semantic vector embedding for an image."""
    now_iso = datetime.utcnow().isoformat()
    data_str = json.dumps(embedding)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO image_embeddings (id, image_id, embedding_provider, embedding_model, embedding_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (embedding_id, image_id, provider, model, data_str, now_iso),
        )
        conn.commit()


def get_all_embeddings(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieve all embeddings for vector search matching."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM image_embeddings")
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["embedding"] = json.loads(d["embedding_data"])
            results.append(d)
        return results


# --- Metrics Summary ---

def get_platform_metrics(db_path: Optional[Path] = None) -> Dict[str, int]:
    """Retrieve aggregate statistics for dashboard and header."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM images")
        total_images = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM image_versions")
        total_versions = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM image_versions WHERE is_original = 0")
        total_edits = cursor.fetchone()["cnt"]

        cursor.execute("SELECT COUNT(*) as cnt FROM image_reverts")
        total_reverts = cursor.fetchone()["cnt"]

        return {
            "total_images": total_images,
            "total_versions": total_versions,
            "total_edits": total_edits,
            "total_reverts": total_reverts,
        }
