"""Download and verify the NASA C-MAPSS dataset (FR-01, FR-16).

The primary source is the NASA data.gov legacy zip; if it is unreachable or
its checksum does not match, the script falls back to the PCoE Zenodo mirror.
The downloaded archive is checksum-verified (md5) before extraction and every
attempt is recorded in ``data/provenance.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import get_settings  # noqa: E402

NASA_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
ZENODO_URL = "https://zenodo.org/api/records/15346912/files/CMAPSSData.zip/content"
EXPECTED_MD5 = "79a22f36e80606c69d0e9e4da5bb2b7a"
DEFAULT_TIMEOUT_SECONDS = 60


def md5_hex(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download(source: str, destination: Path, timeout: int) -> None:
    """Download ``source`` to ``destination``, following redirects."""
    request = urllib.request.Request(source, headers={"User-Agent": "industrial-ml-pipeline/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with destination.open("wb") as out:
            shutil.copyfileobj(response, out)


def _safe_extract(archive: Path, raw_dir: Path) -> None:
    with zipfile.ZipFile(archive) as zip_handle:
        for member in zip_handle.infolist():
            target = (raw_dir / member.filename).resolve()
            if not target.is_relative_to(raw_dir.resolve()):
                raise ValueError(f"unsafe zip entry: {member.filename}")
    with zipfile.ZipFile(archive) as zip_handle:
        zip_handle.extractall(raw_dir)


def fetch_cmapss(
    raw_dir: Path | None = None,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    sources: tuple[str, ...] = (NASA_URL, ZENODO_URL),
    expected_md5: str = EXPECTED_MD5,
) -> Path:
    """Fetch the archive from the first source whose checksum matches."""
    settings = get_settings()
    root = raw_dir or settings.raw_dir
    root.mkdir(parents=True, exist_ok=True)

    archive = root / "CMAPSSData.zip"
    downloads: list[dict[str, object]] = []
    part_file = root / ".CMAPSSData.zip.part"
    downloaded = False
    for source in sources:
        try:
            _download(source, part_file, timeout)
            actual_md5 = md5_hex(part_file)
            downloads.append(
                {
                    "source": source,
                    "ok": actual_md5 == expected_md5,
                    "md5": actual_md5,
                    "bytes": part_file.stat().st_size,
                    "attempted_at": datetime.now(UTC).isoformat(),
                }
            )
            if actual_md5 == expected_md5:
                part_file.replace(archive)
                downloaded = True
                break
        except Exception as exc:  # noqa: BLE001 - a failed mirror must not abort the run
            downloads.append(
                {
                    "source": source,
                    "ok": False,
                    "error": str(exc),
                    "attempted_at": datetime.now(UTC).isoformat(),
                }
            )
        finally:
            if part_file.exists():
                part_file.unlink()

    if not downloaded or not archive.exists():
        provenance_path = root.parent / "provenance.json"
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps({"dataset": "CMAPSS", "downloads": downloads}, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"CMAPSSData.zip could not be downloaded with the expected md5 "
            f"({expected_md5}) from any source. See {provenance_path}"
        )

    _safe_extract(archive, root)
    provenance = {
        "dataset": "CMAPSS",
        "expected_md5": expected_md5,
        "archive": str(archive),
        "downloads": downloads,
        "extracted_at": datetime.now(UTC).isoformat(),
    }
    provenance_path = root.parent / "provenance.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify C-MAPSS data.")
    parser.add_argument("--raw-dir", type=Path, default=None, help="raw data directory")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    archive = fetch_cmapss(raw_dir=args.raw_dir, timeout=args.timeout)
    print(f"verified archive: {archive}")


if __name__ == "__main__":
    main()
