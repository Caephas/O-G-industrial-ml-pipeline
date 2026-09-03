"""Fetch script tests: checksum verification, fallback, extraction."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts.fetch_cmapss import fetch_cmapss, md5_hex


def _make_zip(path: Path, content: str = "synthetic fixture archive") -> str:
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr("CMAPSSData/readme.txt", content)
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_fetch_success_writes_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.zip"
    expected_md5 = _make_zip(source)
    raw_dir = tmp_path / "raw"

    archive = fetch_cmapss(
        raw_dir=raw_dir,
        sources=(source.as_uri(),),
        expected_md5=expected_md5,
    )

    assert archive.exists()
    assert md5_hex(archive) == expected_md5
    extracted = raw_dir / "CMAPSSData" / "readme.txt"
    assert extracted.read_text(encoding="utf-8") == "synthetic fixture archive"
    provenance = (raw_dir.parent / "provenance.json").read_text(encoding="utf-8")
    assert source.as_uri() in provenance


def test_fetch_falls_back_on_mismatched_checksum(tmp_path: Path) -> None:
    bad_source = tmp_path / "bad.zip"
    good_source = tmp_path / "good.zip"
    bad_md5 = _make_zip(bad_source, content="corrupted archive")
    _make_zip(good_source, content="valid archive")
    good_md5 = hashlib.md5(good_source.read_bytes()).hexdigest()
    raw_dir = tmp_path / "raw"

    archive = fetch_cmapss(
        raw_dir=raw_dir,
        sources=(bad_source.as_uri(), good_source.as_uri()),
        expected_md5=good_md5,
    )
    assert md5_hex(archive) == good_md5
    assert bad_md5 != good_md5


def test_fetch_raises_when_all_sources_fail(tmp_path: Path) -> None:
    source = tmp_path / "wrong.zip"
    _make_zip(source, content="wrong content")
    raw_dir = tmp_path / "raw"

    with pytest.raises(RuntimeError):
        fetch_cmapss(raw_dir=raw_dir, sources=(source.as_uri(),), expected_md5="0" * 32)
    assert not (raw_dir / ".CMAPSSData.zip.part").exists()
