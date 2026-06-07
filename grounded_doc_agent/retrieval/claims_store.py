from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from grounded_doc_agent.models import Claim, ClaimConflict


class ClaimsStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    value TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    section_path TEXT NOT NULL,
                    source_label TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject);
                CREATE TABLE IF NOT EXISTS conflicts (
                    subject TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    claim_ids TEXT NOT NULL
                );
                """
            )

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM claims")
            conn.execute("DELETE FROM conflicts")

    def delete_claims_for_doc(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM claims WHERE doc_id = ?", (doc_id,))

    def list_all_claims(self) -> list[Claim]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM claims").fetchall()
        return [self._row_to_claim(row) for row in rows]

    def rebuild_conflicts(self) -> list[ClaimConflict]:
        from grounded_doc_agent.ingestion.claims import detect_conflicts

        claims = self.list_all_claims()
        conflicts = detect_conflicts(claims)
        with self._connect() as conn:
            conn.execute("DELETE FROM conflicts")
        self.upsert_conflicts(conflicts)
        return conflicts

    def upsert_claims(self, claims: list[Claim]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO claims
                (claim_id, subject, value, doc_id, chunk_id, section_path, source_label)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        claim.claim_id,
                        claim.subject,
                        claim.value,
                        claim.doc_id,
                        claim.chunk_id,
                        claim.section_path,
                        claim.source_label,
                    )
                    for claim in claims
                ],
            )

    def upsert_conflicts(self, conflicts: list[ClaimConflict]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO conflicts (subject, description, claim_ids)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        conflict.subject,
                        conflict.description,
                        json.dumps([claim.claim_id for claim in conflict.claims]),
                    )
                    for conflict in conflicts
                ],
            )

    def get_claims_by_subject(self, subject: str) -> list[Claim]:
        normalized = subject.lower().replace(" ", "_")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM claims WHERE subject = ? OR subject LIKE ?",
                (normalized, f"%{normalized}%"),
            ).fetchall()
        return [self._row_to_claim(row) for row in rows]

    def search_claims(self, query: str) -> list[Claim]:
        tokens = [token for token in query.lower().split() if len(token) > 3]
        if not tokens:
            return []
        clauses = " OR ".join(["subject LIKE ? OR value LIKE ?"] * len(tokens))
        params: list[str] = []
        for token in tokens:
            params.extend([f"%{token}%", f"%{token}%"])
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM claims WHERE {clauses} LIMIT 20",
                params,
            ).fetchall()
        return [self._row_to_claim(row) for row in rows]

    def get_conflicts_for_subjects(self, subjects: set[str]) -> list[ClaimConflict]:
        if not subjects:
            return []
        placeholders = ",".join("?" for _ in subjects)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM conflicts WHERE subject IN ({placeholders})",
                list(subjects),
            ).fetchall()
        conflicts: list[ClaimConflict] = []
        for row in rows:
            claim_ids = json.loads(row["claim_ids"])
            claims = self.get_claims_by_ids(claim_ids)
            conflicts.append(
                ClaimConflict(
                    subject=row["subject"],
                    claims=claims,
                    description=row["description"],
                )
            )
        return conflicts

    def get_claims_by_ids(self, claim_ids: list[str]) -> list[Claim]:
        if not claim_ids:
            return []
        placeholders = ",".join("?" for _ in claim_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM claims WHERE claim_id IN ({placeholders})",
                claim_ids,
            ).fetchall()
        return [self._row_to_claim(row) for row in rows]

    def list_conflicts(self) -> list[ClaimConflict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM conflicts").fetchall()
        conflicts: list[ClaimConflict] = []
        for row in rows:
            claim_ids = json.loads(row["claim_ids"])
            claims = self.get_claims_by_ids(claim_ids)
            conflicts.append(
                ClaimConflict(
                    subject=row["subject"],
                    claims=claims,
                    description=row["description"],
                )
            )
        return conflicts

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> Claim:
        return Claim(
            claim_id=row["claim_id"],
            subject=row["subject"],
            value=row["value"],
            doc_id=row["doc_id"],
            chunk_id=row["chunk_id"],
            section_path=row["section_path"],
            source_label=row["source_label"],
        )
