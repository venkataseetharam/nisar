"""
Extract identification metadata + backscatter QA stats from a NISAR L2 HDF5
granule into a JSON sidecar consumable by team/projects/map/nisar-line.html.

The bundled hdf5.min.js (jsfive) has a B-tree v2 traversal bug that triggers
"Offset is outside the bounds of the DataView" when descending into NISAR's
identification group, so the browser falls back on this sidecar instead.

Run:
    python3 extract_h5_metadata.py
    python3 extract_h5_metadata.py --h5 path/to/file.h5 --out path/to/file.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import h5py
except ImportError:
    print("[h5-meta] ERROR: h5py is required. Install with `pip install h5py`.", file=sys.stderr)
    sys.exit(2)


# Fields exported from /science/LSAR/identification
IDENTIFICATION_FIELDS = [
    "granuleId",
    "instrumentName",
    "missionId",
    "trackNumber",
    "frameNumber",
    "absoluteOrbitNumber",
    "orbitPassDirection",
    "lookDirection",
    "productDoi",
    "productSpecificationVersion",
    "processingCenter",
    "processingDateTime",
    "productType",
    "radarBand",
    "zeroDopplerStartTime",
    "zeroDopplerEndTime",
    "boundingPolygon",
]

# QA backscatter statistics per polarization
QA_STAT_FIELDS = ["min_value", "max_value", "mean_value", "sample_stddev"]


def _decode(value):
    """Return a JSON-friendly scalar from an h5py value (bytes -> str)."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        decoded = value.tolist()
    else:
        decoded = value
    if isinstance(decoded, list) and len(decoded) == 1:
        decoded = decoded[0]
    if isinstance(decoded, bytes):
        return decoded.decode("utf-8", errors="replace")
    return decoded


def parse_bounding_polygon(wkt: str) -> list[list[float]]:
    """Parse 'POLYGON ((lon lat [z], lon lat [z], …))' -> [[lat, lon], …]."""
    if not wkt or "POLYGON" not in wkt:
        return []
    try:
        start = wkt.index("((") + 2
        end = wkt.index("))", start)
    except ValueError:
        return []
    points: list[list[float]] = []
    for pt in wkt[start:end].split(","):
        parts = pt.strip().split()
        if len(parts) < 2:
            continue
        try:
            lon, lat = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        points.append([lat, lon])
    return points


def extract_metadata(h5_path: Path) -> dict:
    out: dict = {
        "source_file": h5_path.name,
        "identification": {},
        "boundingPolygonLatLon": [],
        "qaStats": {},
    }
    with h5py.File(h5_path, "r") as h5:
        ident_group = h5.get("science/LSAR/identification")
        if ident_group is None:
            raise RuntimeError("science/LSAR/identification group not found in HDF5 file")

        for name in IDENTIFICATION_FIELDS:
            ds = ident_group.get(name)
            if ds is None:
                continue
            out["identification"][name] = _decode(ds[()])

        wkt = out["identification"].get("boundingPolygon", "")
        out["boundingPolygonLatLon"] = parse_bounding_polygon(wkt)

        qa_group = h5.get("science/LSAR/QA/data/frequencyA")
        if qa_group is not None:
            for pol_name in qa_group:
                pol_group = qa_group.get(pol_name)
                if not isinstance(pol_group, h5py.Group):
                    continue
                stats = {}
                for stat in QA_STAT_FIELDS:
                    ds = pol_group.get(stat)
                    if ds is not None:
                        stats[stat] = _decode(ds[()])
                if stats:
                    out["qaStats"][pol_name] = stats

    return out


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--h5",
        default=str(here.parent / "NISAR_L2.h5"),
        help="Path to the NISAR L2 HDF5 file (default: ../NISAR_L2.h5)",
    )
    parser.add_argument(
        "--out",
        default=str(here.parent / "nisar_l2_metadata.json"),
        help="Output JSON path (default: ../nisar_l2_metadata.json)",
    )
    args = parser.parse_args(argv)

    h5_path = Path(args.h5).resolve()
    out_path = Path(args.out).resolve()

    if not h5_path.exists():
        print(f"[h5-meta] ERROR: {h5_path} does not exist", file=sys.stderr)
        return 1

    metadata = extract_metadata(h5_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metadata, indent=2))

    ident = metadata["identification"]
    poly = metadata["boundingPolygonLatLon"]
    print(f"[h5-meta] wrote {out_path}")
    print(f"  granule:    {ident.get('granuleId', '—')}")
    print(f"  instrument: {ident.get('instrumentName', '—')}")
    print(f"  acquired:   {ident.get('zeroDopplerStartTime', '—')}")
    print(f"  polygon:    {len(poly)} vertices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
