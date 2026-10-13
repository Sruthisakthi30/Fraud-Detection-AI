"""
FraudShield AI - Investigation Cases Service (SQLite Persistence)
Provides case management for flagged transactions without claiming definitive fraud.
"""
import sqlite3
import logging
from pathlib import Path
from typing import Any
from src.utils import utc_now_iso

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "results" / "cases.db"


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_cases_db() -> None:
    """Initialize cases table if not exists."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                amount REAL,
                time REAL,
                risk_level TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'Open',
                flagged_by TEXT NOT NULL DEFAULT 'FraudShield AI Model',
                notes TEXT,
                features_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()


def get_all_cases(status: str | None = None, risk_level: str | None = None, search: str | None = None) -> list[dict[str, Any]]:
    """Retrieve cases with optional filtering."""
    init_cases_db()
    query = "SELECT * FROM cases WHERE 1=1"
    params: list[Any] = []

    if status and status != "All":
        query += " AND status = ?"
        params.append(status)

    if risk_level and risk_level != "All":
        query += " AND risk_level = ?"
        params.append(risk_level)

    if search:
        query += " AND (case_id LIKE ? OR title LIKE ? OR notes LIKE ?)"
        term = f"%{search}%"
        params.extend([term, term, term])

    query += " ORDER BY id DESC"

    with _get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def create_case(data: dict[str, Any]) -> dict[str, Any]:
    """Create a new investigation case."""
    init_cases_db()
    import uuid
    short_code = str(uuid.uuid4())[:8].upper()
    case_id = f"CASE-{short_code}"
    title = data.get("title", f"Investigation for Transaction {case_id}")
    amount = float(data.get("amount", 0.0))
    time_val = float(data.get("time", 0.0))
    risk_level = data.get("risk_level", "Medium")
    status = data.get("status", "Open")
    flagged_by = data.get("flagged_by", "FraudShield AI Model")
    notes = data.get("notes", "Auto-generated investigation case pending analyst review.")
    features_json = data.get("features_json", "{}")
    now = utc_now_iso()

    with _get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO cases (case_id, title, amount, time, risk_level, status, flagged_by, notes, features_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (case_id, title, amount, time_val, risk_level, status, flagged_by, notes, features_json, now, now))
        conn.commit()
        new_id = cur.lastrowid

    return {
        "id": new_id,
        "case_id": case_id,
        "title": title,
        "amount": amount,
        "time": time_val,
        "risk_level": risk_level,
        "status": status,
        "flagged_by": flagged_by,
        "notes": notes,
        "created_at": now,
        "updated_at": now,
    }


def update_case(case_id_or_pk: int | str, data: dict[str, Any]) -> dict[str, Any] | None:
    """Update case status or notes."""
    init_cases_db()
    now = utc_now_iso()
    updates = []
    params: list[Any] = []

    if "status" in data:
        updates.append("status = ?")
        params.append(data["status"])
    if "notes" in data:
        updates.append("notes = ?")
        params.append(data["notes"])
    if "risk_level" in data:
        updates.append("risk_level = ?")
        params.append(data["risk_level"])
    if "title" in data:
        updates.append("title = ?")
        params.append(data["title"])

    if not updates:
        return None

    updates.append("updated_at = ?")
    params.append(now)

    col = "id" if isinstance(case_id_or_pk, int) or str(case_id_or_pk).isdigit() else "case_id"
    params.append(case_id_or_pk)

    sql = f"UPDATE cases SET {', '.join(updates)} WHERE {col} = ?"

    with _get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        if cur.rowcount == 0:
            return None
        row = conn.execute(f"SELECT * FROM cases WHERE {col} = ?", (case_id_or_pk,)).fetchone()
        return dict(row) if row else None


def delete_case(case_id_or_pk: int | str) -> bool:
    """Delete a case by id or case_id."""
    init_cases_db()
    col = "id" if isinstance(case_id_or_pk, int) or str(case_id_or_pk).isdigit() else "case_id"
    with _get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM cases WHERE {col} = ?", (case_id_or_pk,))
        conn.commit()
        return cur.rowcount > 0
