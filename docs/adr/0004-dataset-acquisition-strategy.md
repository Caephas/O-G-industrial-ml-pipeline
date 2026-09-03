# ADR 0004: Checksummed Multi-Source Dataset Acquisition

- Status: Accepted
- Date: 2026-09-03

## Context

The repo must acquire the NASA C-MAPSS archive reproducibly (FR-01, FR-16).
The NASA data.gov download endpoint redirects to short-lived signed storage
URLs and is not always reachable, so a single hardcoded URL would make the
setup fragile.

## Decision

`scripts/fetch_cmapss.py` downloads `CMAPSSData.zip`, verifies its md5
(`79a22f36e80606c69d0e9e4da5bb2b7a`, matching the PCoE Zenodo mirror),
extracts it safely, and records every attempt in `data/provenance.json`.
Sources are tried in order: NASA data.gov first, then the Zenodo mirror.
Processed outputs are never committed to the repository; they are
reproducible from the checksummed archive.

## Consequences

- A failed or tampered download fails loudly and records which source failed.
- The 12 MB archive is fetched once and cached in the gitignored `data/raw/`
  directory.
- Dataset provenance is auditable per run and feeds artifact manifests later.
