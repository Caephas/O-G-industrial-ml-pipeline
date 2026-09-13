"""SQLite-backed model registry implementing the frozen protocol (FR-10).

Artifacts remain self-describing joblib + manifest files; the SQLite index
adds transactional stage transitions and a queryable history. See ADR 0009.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from pipeline.config import ModelFamily, ModelStage, get_settings
from pipeline.schemas.artifact import ArtifactManifest
from pipeline.schemas.registry import (
    ModelRegistry,
    RegistryEntry,
    RegistryError,
    StageTransition,
)

_ALLOWED_TRANSITIONS: set[tuple[str, str]] = {
    ("staging", "production"),
    ("staging", "archived"),
    ("production", "archived"),
}


class SqliteRegistry(ModelRegistry):
    """Default registry backend: SQLite index over self-describing artifacts."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or get_settings().registry_dir / "registry.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    family TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    subset TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    registered_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    artifact_id TEXT NOT NULL,
                    from_stage TEXT,
                    to_stage TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    occurred_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_family ON artifacts(family, stage);
                CREATE INDEX IF NOT EXISTS idx_transitions_artifact ON transitions(artifact_id);
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def register(self, manifest: ArtifactManifest) -> RegistryEntry:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT artifact_id FROM artifacts WHERE artifact_id = ?",
                (manifest.artifact_id,),
            ).fetchone()
            if existing:
                raise RegistryError(f"artifact already registered: {manifest.artifact_id}")
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM artifacts WHERE family = ?",
                (manifest.family,),
            ).fetchone()
            version = int(row[0])
            staged = manifest.model_copy(update={"version": version, "stage": "staging"})
            connection.execute(
                """
                INSERT INTO artifacts (
                    artifact_id, family, version, stage, subset, run_id,
                    artifact_path, manifest_json, registered_at
                ) VALUES (?, ?, ?, 'staging', ?, ?, ?, ?, ?)
                """,
                (
                    staged.artifact_id,
                    staged.family,
                    staged.version,
                    staged.subset,
                    staged.run_id,
                    staged.artifact_path,
                    staged.model_dump_json(),
                    self._now(),
                ),
            )
            self._record_transition(
                connection,
                staged.artifact_id,
                None,
                "staging",
                "registered",
            )
        return self.get(manifest.artifact_id)

    @staticmethod
    def _record_transition(
        connection: sqlite3.Connection,
        artifact_id: str,
        from_stage: ModelStage | None,
        to_stage: ModelStage,
        reason: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO transitions (artifact_id, from_stage, to_stage, reason, occurred_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (artifact_id, from_stage, to_stage, reason, SqliteRegistry._now()),
        )

    def _transition(self, artifact_id: str, to_stage: ModelStage, reason: str) -> RegistryEntry:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
            if row is None:
                raise RegistryError(f"unknown artifact: {artifact_id}")
            current = row["stage"]
            if (current, to_stage) not in _ALLOWED_TRANSITIONS:
                raise RegistryError(
                    f"illegal transition {current} -> {to_stage} for {artifact_id}"
                )
            connection.execute(
                "UPDATE artifacts SET stage = ? WHERE artifact_id = ?",
                (to_stage, artifact_id),
            )
            self._record_transition(connection, artifact_id, current, to_stage, reason)

            # Promoting a staging model supersedes the current production model
            # of the same family, keeping exactly one production artifact.
            if to_stage == "production" and current == "staging":
                prior = connection.execute(
                    """
                    SELECT artifact_id FROM artifacts
                    WHERE family = ? AND stage = 'production' AND artifact_id != ?
                    """,
                    (row["family"], artifact_id),
                ).fetchone()
                if prior is not None:
                    connection.execute(
                        "UPDATE artifacts SET stage = 'archived' WHERE artifact_id = ?",
                        (prior["artifact_id"],),
                    )
                    self._record_transition(
                        connection,
                        prior["artifact_id"],
                        "production",
                        "archived",
                        f"superseded by {artifact_id}",
                    )
        return self.get(artifact_id)

    def promote(self, artifact_id: str, *, reason: str = "shadow gate passed") -> RegistryEntry:
        return self._transition(artifact_id, "production", reason)

    def archive(self, artifact_id: str, *, reason: str) -> RegistryEntry:
        return self._transition(artifact_id, "archived", reason)

    def list(
        self,
        family: ModelFamily | None = None,
        stage: ModelStage | None = None,
    ) -> list[RegistryEntry]:
        query = "SELECT * FROM artifacts WHERE 1 = 1"
        params: list[str] = []
        if family is not None:
            query += " AND family = ?"
            params.append(family)
        if stage is not None:
            query += " AND stage = ?"
            params.append(stage)
        query += " ORDER BY family, version"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._entry_from_row(row) for row in rows]

    def get(self, artifact_id: str) -> RegistryEntry:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise RegistryError(f"unknown artifact: {artifact_id}")
        return self._entry_from_row(row)

    def get_production(self, family: ModelFamily) -> RegistryEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM artifacts
                WHERE family = ? AND stage = 'production'
                ORDER BY version DESC LIMIT 1
                """,
                (family,),
            ).fetchone()
        return self._entry_from_row(row) if row is not None else None

    def history(self, artifact_id: str) -> list[StageTransition]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM transitions WHERE artifact_id = ?
                ORDER BY id
                """,
                (artifact_id,),
            ).fetchall()
        return [
            StageTransition(
                artifact_id=row["artifact_id"],
                from_stage=row["from_stage"],
                to_stage=row["to_stage"],
                reason=row["reason"],
                occurred_at=datetime.fromisoformat(row["occurred_at"]),
            )
            for row in rows
        ]

    def _entry_from_row(self, row: sqlite3.Row) -> RegistryEntry:
        manifest = ArtifactManifest.model_validate_json(row["manifest_json"])
        manifest = manifest.model_copy(update={"stage": row["stage"]})
        return RegistryEntry(
            manifest=manifest,
            registered_at=datetime.fromisoformat(row["registered_at"]),
            history=self.history(row["artifact_id"]),
        )

    def manifest_json(self, artifact_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT manifest_json FROM artifacts WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise RegistryError(f"unknown artifact: {artifact_id}")
        return row["manifest_json"]
