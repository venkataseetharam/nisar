"""
NISAR Subsidence Data Pipeline
================================
Queries NASA ASF DAAC for NISAR displacement products, extracts subsidence
rates per location, and writes a CSV enriched with US state/county/ZIP.

Usage:
    python fetch_nisar.py [--bbox "lon_min,lat_min,lon_max,lat_max"] [--output path/to/out.csv]

Auth (NASA Earthdata):
    Set env vars EARTHDATA_USER and EARTHDATA_PASS before running,
    or create ~/.netrc manually.

Requirements: see requirements.txt
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import textwrap
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_CSV = Path(__file__).parent.parent / "data" / "nisar_subsidence.csv"

# US Census TIGER shapefiles (downloaded at runtime if missing)
CENSUS_BASE = "https://www2.census.gov/geo/tiger/TIGER2023"
STATES_URL  = f"{CENSUS_BASE}/STATE/tl_2023_us_state.zip"
COUNTIES_URL= f"{CENSUS_BASE}/COUNTY/tl_2023_us_county.zip"
ZCTA_URL    = f"{CENSUS_BASE}/ZCTA520/tl_2023_us_zcta520.zip"

# ASF CMR endpoint
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"

# Hardcoded fallback rows (used when no real NISAR data is found for a region)
FALLBACK_ROWS = [
    {
        "lat": 29.76, "lon": -95.37,
        "subsidence_value": 1.0, "subsidence_unit": "cm/year",
        "location_name": "Houston, TX", "state": "Texas",
        "county": "Harris County", "zip": "77002",
        "season": "annual", "source_date": "2025-Q4",
        "source": "reference",
        "ref_date": "", "sec_date": "", "baseline_days": 12,
    },
    {
        "lat": 35.69, "lon": 51.39,
        "subsidence_value": 33.0, "subsidence_unit": "cm/year",
        "location_name": "Tehran, Iran", "state": "", "county": "", "zip": "",
        "season": "annual", "source_date": "2025-Q4",
        "source": "reference",
        "ref_date": "", "sec_date": "", "baseline_days": 12,
    },
    {
        "lat": -6.13, "lon": 106.84,
        "subsidence_value": 27.9, "subsidence_unit": "cm/year",
        "location_name": "North Jakarta, Indonesia", "state": "", "county": "", "zip": "",
        "season": "annual", "source_date": "2025-Q4",
        "source": "reference",
        "ref_date": "", "sec_date": "", "baseline_days": 12,
    },
]

CSV_FIELDS = [
    "lat", "lon", "subsidence_value", "subsidence_unit",
    "location_name", "state", "county", "zip",
    "season", "source_date", "source",
    "ref_date", "sec_date", "baseline_days",
]


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def write_netrc(user: str, password: str) -> None:
    """Write Earthdata credentials to ~/.netrc so requests/asf_search auth works."""
    netrc_path = Path.home() / ".netrc"
    entry = textwrap.dedent(f"""\
        machine urs.earthdata.nasa.gov
            login {user}
            password {password}
        machine cmr.earthdata.nasa.gov
            login {user}
            password {password}
    """)
    existing = netrc_path.read_text() if netrc_path.exists() else ""
    if "urs.earthdata.nasa.gov" not in existing:
        with netrc_path.open("a") as f:
            f.write(entry)
        netrc_path.chmod(0o600)


def setup_auth() -> None:
    user = os.environ.get("EARTHDATA_USER", "")
    password = os.environ.get("EARTHDATA_PASS", "")
    if user and password:
        write_netrc(user, password)
    else:
        print("WARNING: EARTHDATA_USER/EARTHDATA_PASS not set. Will try ~/.netrc.")


# ---------------------------------------------------------------------------
# ASF CMR query for NISAR products
# ---------------------------------------------------------------------------

def query_nisar_products(bbox: str, start: str, end: str) -> list[dict]:
    """
    Query NASA CMR for NISAR DISP-S1 (displacement) granules in bbox.
    Returns a list of granule metadata dicts.
    """
    params = {
        "short_name": "NISAR_L2_GUNW_BETA_V1",  # Geocoded Unwrapped Interferogram (Beta)
        "bounding_box": bbox,               # "lon_min,lat_min,lon_max,lat_max"
        "temporal": f"{start},{end}",
        "page_size": 50,
        "page_num": 1,
    }
    print(f"Querying CMR for NISAR products: bbox={bbox}, temporal={start} to {end}")
    resp = requests.get(CMR_URL, params=params, timeout=30)
    resp.raise_for_status()
    entries = resp.json().get("feed", {}).get("entry", [])
    print(f"Found {len(entries)} granule(s).")
    return entries


def download_granule(granule: dict, dest_dir: Path) -> Path | None:
    """Download the first HDF5 link from a granule entry. Returns local path."""
    links = [
        lnk["href"] for lnk in granule.get("links", [])
        if lnk.get("href", "").endswith(".h5")
    ]
    if not links:
        return None
    url = links[0]
    fname = dest_dir / url.split("/")[-1]
    print(f"Downloading {fname.name} …")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with fname.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return fname


# ---------------------------------------------------------------------------
# HDF5 extraction
# ---------------------------------------------------------------------------

# NISAR L-band radar wavelength (metres). LOS displacement from unwrapped
# phase:  d = -(lambda / 4*pi) * phi
NISAR_WAVELENGTH_M = 0.2422

# ~100 km grid spacing in degrees (1 deg latitude ~= 111 km)
GRID_DEG = 0.9


def _first_existing(h5: h5py.File, candidates: list[str]):
    """Return the first dataset path that exists in the HDF5 file, else None."""
    for path in candidates:
        if path in h5:
            return path
    return None


def _read_dates(h5: h5py.File) -> tuple[str, str, int]:
    """
    Pull the reference / secondary acquisition dates and temporal baseline
    (days) from a GUNW granule. Falls back gracefully if attrs differ.
    """
    id_grp = "/science/LSAR/identification"

    def _get(name, default=""):
        try:
            v = h5[f"{id_grp}/{name}"][()]
            return v.decode() if isinstance(v, bytes) else str(v)
        except Exception:
            return default

    ref = _get("referenceZeroDopplerStartTime") or _get("referenceDatetime")
    sec = _get("secondaryZeroDopplerStartTime") or _get("secondaryDatetime")
    ref_d, sec_d = ref[:10], sec[:10]
    try:
        d0 = datetime.strptime(ref_d, "%Y-%m-%d")
        d1 = datetime.strptime(sec_d, "%Y-%m-%d")
        baseline = abs((d1 - d0).days)
    except Exception:
        baseline = 12  # nominal NISAR repeat
    return ref_d, sec_d, baseline


def extract_displacement(h5_path: Path) -> tuple[pd.DataFrame | None, dict]:
    """
    Read a NISAR GUNW (Geocoded Unwrapped Interferogram) HDF5 file and
    convert unwrapped phase -> line-of-sight elevation change (mm) over the
    interferogram's 6-12 day interval.

    Returns (DataFrame[lat, lon, disp_mm], meta) where meta carries the
    reference/secondary dates and temporal baseline in days.
    """
    meta = {"ref_date": "", "sec_date": "", "baseline_days": 12}
    try:
        with h5py.File(h5_path, "r") as f:
            meta["ref_date"], meta["sec_date"], meta["baseline_days"] = _read_dates(f)

            base = "/science/LSAR/GUNW/grids/frequencyA/unwrappedInterferogram"
            # Polarisation sub-group varies (HH / VV); pick the first present.
            phase_path = _first_existing(f, [
                f"{base}/HH/unwrappedPhase",
                f"{base}/VV/unwrappedPhase",
                f"{base}/unwrappedPhase",
            ])
            coh_path = _first_existing(f, [
                f"{base}/HH/coherenceMagnitude",
                f"{base}/VV/coherenceMagnitude",
                f"{base}/coherenceMagnitude",
            ])
            if phase_path is None:
                print(f"  no unwrappedPhase in {h5_path.name}")
                return None, meta

            phase = f[phase_path][:].astype("float64")

            # Coordinate axes live alongside the grid (xCoordinates /
            # yCoordinates in the granule's projected CRS, plus an EPSG code).
            grid = f[f"{base}/HH"] if f"{base}/HH" in f else f[base]
            x = grid["xCoordinates"][:] if "xCoordinates" in grid else f[f"{base}/xCoordinates"][:]
            y = grid["yCoordinates"][:] if "yCoordinates" in grid else f[f"{base}/yCoordinates"][:]

            lon2d, lat2d = _to_lonlat(f, x, y, base)

            # phase -> LOS displacement in mm
            disp_mm = -(NISAR_WAVELENGTH_M / (4.0 * np.pi)) * phase * 1000.0

            lat_flat = lat2d.ravel()
            lon_flat = lon2d.ravel()
            val_flat = disp_mm.ravel()

            mask = np.isfinite(val_flat) & (val_flat != 0)
            if coh_path is not None:
                coh = f[coh_path][:].astype("float64").ravel()
                mask &= np.isfinite(coh) & (coh > 0.3)   # drop low-coherence noise

            return pd.DataFrame({
                "lat": lat_flat[mask],
                "lon": lon_flat[mask],
                "disp_mm": val_flat[mask],
            }), meta
    except Exception as exc:
        print(f"ERROR reading {h5_path.name}: {exc}")
        return None, meta


def _to_lonlat(h5: h5py.File, x: np.ndarray, y: np.ndarray, base: str):
    """
    Build 2-D lon/lat grids from projected x/y coordinates. GUNW grids are
    UTM/polar-stereo; reproject via the granule's EPSG code. If already
    geographic (EPSG:4326) just meshgrid.
    """
    epsg = 4326
    try:
        proj = h5[f"{base}/projection"]
        epsg = int(proj.attrs.get("epsg_code", proj[()] if proj.shape == () else 4326))
    except Exception:
        pass

    xx, yy = np.meshgrid(x, y)
    if epsg == 4326:
        return xx, yy

    from pyproj import Transformer
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    return lon, lat


def decimate_to_grid(df: pd.DataFrame, grid_deg: float = GRID_DEG) -> pd.DataFrame:
    """
    Reduce a dense raster to one representative point per ~100 km cell
    (median displacement). Keeps the output continent-wide but tiny.
    """
    if df.empty:
        return df
    g = df.copy()
    g["cell_lat"] = (np.round(g["lat"] / grid_deg) * grid_deg).round(3)
    g["cell_lon"] = (np.round(g["lon"] / grid_deg) * grid_deg).round(3)
    agg = (
        g.groupby(["cell_lat", "cell_lon"], as_index=False)
         .agg(disp_mm=("disp_mm", "median"), n=("disp_mm", "size"))
         .rename(columns={"cell_lat": "lat", "cell_lon": "lon"})
    )
    return agg[agg["n"] >= 3].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Geospatial enrichment (US state / county / ZIP)
# ---------------------------------------------------------------------------

def load_shapefile(url: str, cache_dir: Path) -> gpd.GeoDataFrame:
    """Download and cache a TIGER shapefile zip, return GeoDataFrame."""
    fname = cache_dir / url.split("/")[-1]
    if not fname.exists():
        print(f"Downloading shapefile {fname.name} …")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with fname.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return gpd.read_file(f"zip://{fname}")


def enrich_with_geography(df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    """Spatial-join points with state, county, and ZIP shapefiles."""
    states   = load_shapefile(STATES_URL, cache_dir)[["NAME", "geometry"]].rename(columns={"NAME": "state"})
    counties = load_shapefile(COUNTIES_URL, cache_dir)[["NAMELSAD", "geometry"]].rename(columns={"NAMELSAD": "county"})
    zctas    = load_shapefile(ZCTA_URL, cache_dir)[["ZCTA5CE20", "geometry"]].rename(columns={"ZCTA5CE20": "zip"})

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )

    gdf = gpd.sjoin(gdf, states[["state", "geometry"]],   how="left", predicate="within").drop(columns="index_right", errors="ignore")
    gdf = gpd.sjoin(gdf, counties[["county", "geometry"]], how="left", predicate="within").drop(columns="index_right", errors="ignore")
    gdf = gpd.sjoin(gdf, zctas[["zip", "geometry"]],      how="left", predicate="within").drop(columns="index_right", errors="ignore")

    return pd.DataFrame(gdf.drop(columns="geometry"))


# ---------------------------------------------------------------------------
# Season helper
# ---------------------------------------------------------------------------

def date_to_season(date_str: str) -> str:
    """Return meteorological season for a YYYY-MM-DD date string."""
    try:
        month = datetime.strptime(date_str[:10], "%Y-%m-%d").month
        return {12: "Winter", 1: "Winter", 2: "Winter",
                3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer",
                9: "Fall",   10: "Fall",  11: "Fall"}[month]
    except Exception:
        return "annual"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_output_rows(enriched: pd.DataFrame, season: str, meta: dict) -> list[dict]:
    rows = []
    for _, r in enriched.iterrows():
        rows.append({
            "lat":               round(float(r["lat"]), 5),
            "lon":               round(float(r["lon"]), 5),
            "subsidence_value":  round(float(r["disp_mm"]), 2),
            "subsidence_unit":   "mm/interval",
            "location_name":     f"{r.get('county', '')} {r.get('state', '')}".strip() or "Unknown",
            "state":             r.get("state", ""),
            "county":            r.get("county", ""),
            "zip":               r.get("zip", ""),
            "season":            season,
            "source_date":       meta.get("sec_date", ""),
            "source":            "NISAR-GUNW",
            "ref_date":          meta.get("ref_date", ""),
            "sec_date":          meta.get("sec_date", ""),
            "baseline_days":     meta.get("baseline_days", 12),
        })
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and process NISAR subsidence data.")
    parser.add_argument(
        "--bbox",
        default="-125,24,-66,49",
        help='Bounding box "lon_min,lat_min,lon_max,lat_max" (default: contiguous US)',
    )
    parser.add_argument("--output", default=str(OUTPUT_CSV), help="Output CSV path")
    parser.add_argument(
        "--days-back", type=int, default=90,
        help="How many days of NISAR data to query (default: 90)"
    )
    args = parser.parse_args()

    setup_auth()

    end_date   = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days_back)
    start_str  = start_date.strftime("%Y-%m-%dT00:00:00Z")
    end_str    = end_date.strftime("%Y-%m-%dT23:59:59Z")
    season     = date_to_season(end_date.strftime("%Y-%m-%d"))

    granules = query_nisar_products(args.bbox, start_str, end_str)

    all_rows: list[dict] = []
    cache_dir = Path(tempfile.gettempdir()) / "nisar_shapefiles"
    cache_dir.mkdir(exist_ok=True)

    if granules:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for granule in granules[:5]:           # cap at 5 files per run
                h5_path = download_granule(granule, tmp_path)
                if not h5_path:
                    continue
                df_raw, meta = extract_displacement(h5_path)
                if df_raw is None or df_raw.empty:
                    continue
                df_grid = decimate_to_grid(df_raw)        # ~100 km cells
                if df_grid.empty:
                    continue
                df_enriched = enrich_with_geography(df_grid, cache_dir)
                all_rows.extend(build_output_rows(df_enriched, season, meta))
    else:
        print("No NISAR granules found — using hardcoded fallback data.")

    # Always include fallback reference cities
    all_rows.extend(FALLBACK_ROWS)

    write_csv(all_rows, Path(args.output))


if __name__ == "__main__":
    main()
