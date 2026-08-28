"""
Second Brain — SQLite FTS5 Database Module
===========================================
Zero-dependency full-text search using Python's built-in sqlite3.
No vector databases, no embedding models, no pip installs.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "brain.db"


def get_connection() -> sqlite3.Connection:
    """Open a connection with WAL mode for concurrent read safety."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """
    Create the FTS5 virtual table if it doesn't exist.

    Schema:
      id            INTEGER PRIMARY KEY AUTOINCREMENT
      title         TEXT (3-word summary)
      transcript    TEXT (full transcript)
      duration      TEXT (media duration)
      file_type     TEXT (mp3/mp4/m4a)
      timestamp     REAL (epoch seconds)

    FTS5 index on title + transcript for MATCH queries.
    """
    # Main metadata table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS media (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            title         TEXT NOT NULL,
            transcript    TEXT NOT NULL,
            duration      TEXT DEFAULT '',
            file_type     TEXT DEFAULT '',
            timestamp     REAL NOT NULL,
            char_count    INTEGER DEFAULT 0
        )
    """)

    # FTS5 virtual table for full-text search
    # UNINDEXED fields are stored but not searchable
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS media_fts
        USING fts5(
            title,
            transcript,
            timestamp UNINDEXED,
            duration UNINDEXED,
            file_type UNINDEXED,
            content='media',
            content_rowid='id'
        )
    """)

    # Triggers to keep FTS index in sync with the media table
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS media_ai AFTER INSERT ON media BEGIN
            INSERT INTO media_fts(rowid, title, transcript, timestamp, duration, file_type)
            VALUES (new.id, new.title, new.transcript, new.timestamp, new.duration, new.file_type);
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS media_ad AFTER DELETE ON media BEGIN
            INSERT INTO media_fts(media_fts, rowid, title, transcript, timestamp, duration, file_type)
            VALUES ('delete', old.id, old.title, old.transcript, old.timestamp, old.duration, old.file_type);
        END
    """)

    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS media_au AFTER UPDATE ON media BEGIN
            INSERT INTO media_fts(media_fts, rowid, title, transcript, timestamp, duration, file_type)
            VALUES ('delete', old.id, old.title, old.transcript, old.timestamp, old.duration, old.file_type);
            INSERT INTO media_fts(rowid, title, transcript, timestamp, duration, file_type)
            VALUES (new.id, new.title, new.transcript, new.timestamp, new.duration, new.file_type);
        END
    """)

    conn.commit()


def insert_transcript(
    conn: sqlite3.Connection,
    *,
    title: str,
    transcript: str,
    duration: str = "",
    file_type: str = "",
) -> int:
    """Insert a new transcript and return its ID."""
    now = time.time()
    char_count = len(transcript)
    cursor = conn.execute(
        "INSERT INTO media (title, transcript, duration, file_type, timestamp, char_count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (title, transcript, duration, file_type, now, char_count),
    )
    conn.commit()
    return cursor.lastrowid


def search_fts(conn: sqlite3.Connection, query: str, limit: int = 5) -> list[dict]:
    """
    Full-text search using FTS5 MATCH operator.
    Returns the most relevant transcripts ranked by relevance (BM25).
    """
    # Clean query for FTS5 — remove special characters that break MATCH
    clean = query.strip()
    if not clean:
        return []

    # FTS5MATCH requires clean tokens — remove FTS5 special chars
    for ch in ['"', "'", "(", ")", ":", "*", "+", "-", "<", ">", "&", "|", "^", "~", "{", "}"]:
        clean = clean.replace(ch, " ")

    # Split into words and join with OR for broader matching
    words = [w for w in clean.split() if len(w) > 1]
    if not words:
        return []

    fts_query = " OR ".join(words)

    try:
        rows = conn.execute(
            """
            SELECT m.id, m.title, m.transcript, m.duration, m.file_type, m.timestamp,
                   m.char_count, rank
            FROM media_fts
            JOIN media m ON m.id = media_fts.rowid
            WHERE media_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # Fallback: if FTS query fails, do a LIKE search
        like_pattern = f"%{words[0]}%"
        rows = conn.execute(
            """
            SELECT m.id, m.title, m.transcript, m.duration, m.file_type, m.timestamp,
                   m.char_count, 0 as rank
            FROM media m
            WHERE m.title LIKE ? OR m.transcript LIKE ?
            ORDER BY m.timestamp DESC
            LIMIT ?
            """,
            (like_pattern, like_pattern, limit),
        ).fetchall()

    return [
        {
            "id": r[0],
            "title": r[1],
            "transcript": r[2],
            "duration": r[3],
            "file_type": r[4],
            "timestamp": r[5],
            "char_count": r[6],
            "relevance": abs(r[7]) if r[7] else 0,
        }
        for r in rows
    ]


def get_all_media(conn: sqlite3.Connection) -> list[dict]:
    """Return all stored media, newest first."""
    rows = conn.execute(
        "SELECT id, title, transcript, duration, file_type, timestamp, char_count "
        "FROM media ORDER BY timestamp DESC"
    ).fetchall()
    return [
        {
            "id": r[0],
            "title": r[1],
            "transcript_preview": r[2][:200] + "..." if len(r[2]) > 200 else r[2],
            "transcript": r[2],
            "duration": r[3],
            "file_type": r[4],
            "timestamp": r[5],
            "char_count": r[6],
        }
        for r in rows
    ]


def delete_media(conn: sqlite3.Connection, media_id: int) -> bool:
    """Delete a media entry by ID."""
    conn.execute("DELETE FROM media WHERE id = ?", (media_id,))
    conn.commit()
    return True


def get_stats(conn: sqlite3.Connection) -> dict:
    """Return database statistics."""
    count = conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]
    total_chars = conn.execute("SELECT COALESCE(SUM(char_count), 0) FROM media").fetchone()[0]
    types = conn.execute(
        "SELECT file_type, COUNT(*) FROM media GROUP BY file_type"
    ).fetchall()
    return {
        "total_media": count,
        "total_characters": total_chars,
        "by_type": {r[0] or "unknown": r[1] for r in types},
    }
