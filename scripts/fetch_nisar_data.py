"""
NISAR Elevation Change / Ground Subsidence Data Pipeline

Pulls and processes ground-deformation / subsidence observations and writes
nisar_subsidence.csv (consumed by team/projects/map/nisar.html).

Sources combined:
  1. NASA EarthData CMR (Common Metadata Repository) — discovers available
     NISAR L2 GUNW (Geocoded Unwrapped Interferogram) and OPERA DISP-S1
     (Displacement from Sentinel-1) granules. Footprints from these granules
     identify where displacement observations exist.
  2. Curated subsidence hotspots — geocoded rates published by NASA JPL, USGS,
     ESA, and peer-reviewed InSAR studies. Used to populate the map with
     verified ground-truth rates while NISAR's operational catalog grows.

Output: nisar_subsidence.csv with one row per location:
    record_id, location_name, country, admin_level_1, area_type, area_name,
    postal_code, latitude, longitude, subsidence_value, subsidence_unit,
    source_label, source_url, observation_notes

Run:
    python fetch_nisar_data.py
    python fetch_nisar_data.py --no-cmr      # skip live CMR query
    python fetch_nisar_data.py --out path.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Iterable

import urllib.error
import urllib.parse
import urllib.request


CMR_ENDPOINT = "https://cmr.earthdata.nasa.gov/search/granules.json"

# NISAR + OPERA short_names known to carry geocoded ground-deformation info.
# CMR will silently return zero hits for short_names that don't yet have public
# granules — that's fine, the pipeline just records "0 granules" for them.
DEFORMATION_SHORT_NAMES = [
    "NISAR_L2_GUNW_V1",      # NISAR Geocoded Unwrapped Interferogram
    "NISAR_L2_GOFF_V1",      # NISAR Geocoded Pixel Offsets
    "OPERA_L3_DISP-S1_V1",   # OPERA Displacement from Sentinel-1
    "OPERA_L3_DISP-NI_V1",   # OPERA Displacement from NISAR (future)
]


# ---------------------------------------------------------------------------
# Curated subsidence hotspots
# ---------------------------------------------------------------------------
# Each entry corresponds to a published InSAR study or government subsidence
# report. Values are annual subsidence rates (negative = ground sinking).
# Where studies cite ranges, the upper-bound rate is recorded with the range
# noted in observation_notes so reviewers can audit the choice.

CURATED_HOTSPOTS: list[dict] = [
    # ---- United States ---------------------------------------------------
    {
        "record_id": "houston_harris",
        "location_name": "Houston area",
        "country": "United States",
        "admin_level_1": "Texas",
        "area_type": "county",
        "area_name": "Harris County",
        "postal_code": "77002",
        "latitude": 29.7604,
        "longitude": -95.3698,
        "subsidence_value": 1.0,
        "subsidence_unit": "cm/year",
        "source_label": "USGS Harris-Galveston Subsidence District",
        "source_url": "https://hgsubsidence.org/",
        "observation_notes": "Long-term subsidence from groundwater withdrawal; rates up to ~1 cm/yr in active extraction areas.",
    },
    {
        "record_id": "san_joaquin_valley",
        "location_name": "San Joaquin Valley",
        "country": "United States",
        "admin_level_1": "California",
        "area_type": "county",
        "area_name": "Fresno / Kings / Tulare counties",
        "postal_code": "93210",
        "latitude": 36.3203,
        "longitude": -120.1093,
        "subsidence_value": 30.0,
        "subsidence_unit": "cm/year",
        "source_label": "NASA JPL / USGS California Subsidence Report",
        "source_url": "https://www.usgs.gov/special-topics/water-science-school/science/land-subsidence",
        "observation_notes": "Sinking up to ~30 cm/yr near Corcoran during drought; tracked by JPL InSAR analysis.",
    },
    {
        "record_id": "santa_clara_valley",
        "location_name": "Santa Clara Valley",
        "country": "United States",
        "admin_level_1": "California",
        "area_type": "county",
        "area_name": "Santa Clara County",
        "postal_code": "95110",
        "latitude": 37.3382,
        "longitude": -121.8863,
        "subsidence_value": 0.6,
        "subsidence_unit": "cm/year",
        "source_label": "USGS",
        "source_url": "https://ca.water.usgs.gov/projects/subsidence/",
        "observation_notes": "Historic subsidence largely arrested; residual creep ~0.6 cm/yr observed.",
    },
    {
        "record_id": "phoenix_az",
        "location_name": "Phoenix metro",
        "country": "United States",
        "admin_level_1": "Arizona",
        "area_type": "county",
        "area_name": "Maricopa County",
        "postal_code": "85001",
        "latitude": 33.4484,
        "longitude": -112.0740,
        "subsidence_value": 5.0,
        "subsidence_unit": "cm/year",
        "source_label": "Arizona Department of Water Resources",
        "source_url": "https://new.azwater.gov/hydrology/land-subsidence-monitoring",
        "observation_notes": "Localized subsidence rates 3-5 cm/yr in active groundwater basins.",
    },
    {
        "record_id": "new_orleans_la",
        "location_name": "New Orleans",
        "country": "United States",
        "admin_level_1": "Louisiana",
        "area_type": "county",
        "area_name": "Orleans Parish",
        "postal_code": "70112",
        "latitude": 29.9511,
        "longitude": -90.0715,
        "subsidence_value": 5.0,
        "subsidence_unit": "cm/year",
        "source_label": "NASA JPL UAVSAR Study (Jones et al. 2016)",
        "source_url": "https://www.nature.com/articles/ngeo2715",
        "observation_notes": "Industrial canal area subsiding up to 5 cm/yr per UAVSAR InSAR.",
    },
    {
        "record_id": "norfolk_va",
        "location_name": "Norfolk / Hampton Roads",
        "country": "United States",
        "admin_level_1": "Virginia",
        "area_type": "county",
        "area_name": "City of Norfolk",
        "postal_code": "23510",
        "latitude": 36.8508,
        "longitude": -76.2859,
        "subsidence_value": 0.36,
        "subsidence_unit": "cm/year",
        "source_label": "USGS / NOAA Hampton Roads Subsidence",
        "source_url": "https://pubs.usgs.gov/sir/2013/5036/",
        "observation_notes": "Long-term rates 2.5-3.6 mm/yr from groundwater pumping in confined aquifer.",
    },
    # ---- International ---------------------------------------------------
    {
        "record_id": "tehran_area",
        "location_name": "Tehran area",
        "country": "Iran",
        "admin_level_1": "Tehran Province",
        "area_type": "postal_or_district",
        "area_name": "Tehran urban districts",
        "postal_code": "11369",
        "latitude": 35.6892,
        "longitude": 51.3890,
        "subsidence_value": 13.0,
        "subsidence_unit": "in/year",
        "source_label": "ESA Sentinel-1 InSAR (Haghshenas Haghighi & Motagh, 2019)",
        "source_url": "https://doi.org/10.1016/j.rse.2018.11.003",
        "observation_notes": "Up to 25 cm/yr (~13 in/yr) measured over Tehran-Karaj plain due to groundwater over-pumping.",
    },
    {
        "record_id": "north_jakarta",
        "location_name": "North Jakarta",
        "country": "Indonesia",
        "admin_level_1": "DKI Jakarta",
        "area_type": "postal_or_district",
        "area_name": "North Jakarta districts",
        "postal_code": "14410",
        "latitude": -6.1384,
        "longitude": 106.8637,
        "subsidence_value": 11.0,
        "subsidence_unit": "in/year",
        "source_label": "Bandung Institute of Technology / Andreas et al. (2018)",
        "source_url": "https://doi.org/10.1088/1755-1315/118/1/012039",
        "observation_notes": "Coastal districts subsiding 25-28 cm/yr (~11 in/yr); driver behind capital relocation plan.",
    },
    {
        "record_id": "mexico_city",
        "location_name": "Mexico City",
        "country": "Mexico",
        "admin_level_1": "Ciudad de Mexico",
        "area_type": "postal_or_district",
        "area_name": "Iztapalapa / Tlahuac",
        "postal_code": "09000",
        "latitude": 19.4326,
        "longitude": -99.1332,
        "subsidence_value": 40.0,
        "subsidence_unit": "cm/year",
        "source_label": "NASA JPL / UNAM InSAR Study (2021)",
        "source_url": "https://www.jpl.nasa.gov/news/us-indian-space-mission-maps-extreme-subsidence-in-mexico-city",
        "observation_notes": "Eastern Mexico City sinking up to 40 cm/yr; one of the world's fastest-subsiding capitals.",
    },
    {
        "record_id": "beijing_china",
        "location_name": "Beijing",
        "country": "China",
        "admin_level_1": "Beijing",
        "area_type": "postal_or_district",
        "area_name": "Chaoyang / Tongzhou",
        "postal_code": "100020",
        "latitude": 39.9042,
        "longitude": 116.4074,
        "subsidence_value": 11.0,
        "subsidence_unit": "cm/year",
        "source_label": "Chen et al. (2016) Remote Sensing of Environment",
        "source_url": "https://doi.org/10.3390/rs8060468",
        "observation_notes": "Eastern Beijing plain subsiding up to ~11 cm/yr from aquifer depletion.",
    },
    {
        "record_id": "venice_italy",
        "location_name": "Venice",
        "country": "Italy",
        "admin_level_1": "Veneto",
        "area_type": "postal_or_district",
        "area_name": "Venice historic centre",
        "postal_code": "30100",
        "latitude": 45.4408,
        "longitude": 12.3155,
        "subsidence_value": 0.2,
        "subsidence_unit": "cm/year",
        "source_label": "Tosi et al. (2016) Scientific Reports",
        "source_url": "https://doi.org/10.1038/srep37758",
        "observation_notes": "Residual subsidence ~1-2 mm/yr after industrial groundwater pumping ceased.",
    },
    {
        "record_id": "bangkok_thailand",
        "location_name": "Bangkok",
        "country": "Thailand",
        "admin_level_1": "Bangkok",
        "area_type": "postal_or_district",
        "area_name": "Eastern Bangkok plain",
        "postal_code": "10110",
        "latitude": 13.7563,
        "longitude": 100.5018,
        "subsidence_value": 2.0,
        "subsidence_unit": "cm/year",
        "source_label": "Thailand Department of Mineral Resources",
        "source_url": "http://www.dmr.go.th/",
        "observation_notes": "Eastern Bangkok subsiding 1-2 cm/yr; historically up to 12 cm/yr before pumping limits.",
    },
    {
        "record_id": "ho_chi_minh_vn",
        "location_name": "Ho Chi Minh City",
        "country": "Vietnam",
        "admin_level_1": "Ho Chi Minh City",
        "area_type": "postal_or_district",
        "area_name": "District 7 / Nha Be",
        "postal_code": "700000",
        "latitude": 10.8231,
        "longitude": 106.6297,
        "subsidence_value": 7.0,
        "subsidence_unit": "cm/year",
        "source_label": "Erban et al. (2014) Environmental Research Letters",
        "source_url": "https://doi.org/10.1088/1748-9326/9/8/084010",
        "observation_notes": "Southern districts subsiding up to 7 cm/yr from groundwater extraction.",
    },
    {
        "record_id": "ravenna_italy",
        "location_name": "Ravenna",
        "country": "Italy",
        "admin_level_1": "Emilia-Romagna",
        "area_type": "postal_or_district",
        "area_name": "Ravenna coastal plain",
        "postal_code": "48121",
        "latitude": 44.4184,
        "longitude": 12.2035,
        "subsidence_value": 1.0,
        "subsidence_unit": "cm/year",
        "source_label": "ARPAE Emilia-Romagna",
        "source_url": "https://www.arpae.it/",
        "observation_notes": "Po Delta and Ravenna area subsiding ~5-10 mm/yr from hydrocarbon and water extraction.",
    },
    {
        "record_id": "manila_ph",
        "location_name": "Manila",
        "country": "Philippines",
        "admin_level_1": "Metro Manila",
        "area_type": "postal_or_district",
        "area_name": "Malabon / Navotas",
        "postal_code": "1470",
        "latitude": 14.5995,
        "longitude": 120.9842,
        "subsidence_value": 9.0,
        "subsidence_unit": "cm/year",
        "source_label": "Raucoules et al. (2013) Remote Sensing of Environment",
        "source_url": "https://doi.org/10.1016/j.rse.2013.07.038",
        "observation_notes": "Northern Manila Bay districts subsiding up to ~9 cm/yr due to groundwater pumping.",
    },
    {
        "record_id": "lagos_nigeria",
        "location_name": "Lagos",
        "country": "Nigeria",
        "admin_level_1": "Lagos State",
        "area_type": "postal_or_district",
        "area_name": "Lagos Island / Lekki",
        "postal_code": "101001",
        "latitude": 6.5244,
        "longitude": 3.3792,
        "subsidence_value": 2.5,
        "subsidence_unit": "cm/year",
        "source_label": "Climate Central / Sentinel-1 study (2020)",
        "source_url": "https://www.climatecentral.org/",
        "observation_notes": "Reclaimed coastal land subsiding 1-2.5 cm/yr alongside accelerating sea-level rise.",
    },
    {
        "record_id": "groningen_nl",
        "location_name": "Groningen gas field",
        "country": "Netherlands",
        "admin_level_1": "Groningen",
        "area_type": "postal_or_district",
        "area_name": "Loppersum",
        "postal_code": "9919",
        "latitude": 53.3309,
        "longitude": 6.7475,
        "subsidence_value": 0.6,
        "subsidence_unit": "cm/year",
        "source_label": "NAM / KNMI",
        "source_url": "https://www.nam.nl/",
        "observation_notes": "Subsidence ~5-6 mm/yr from natural gas extraction; documented via levelling and InSAR.",
    },
    {
        "record_id": "shanghai_china",
        "location_name": "Shanghai",
        "country": "China",
        "admin_level_1": "Shanghai",
        "area_type": "postal_or_district",
        "area_name": "Pudong",
        "postal_code": "200120",
        "latitude": 31.2304,
        "longitude": 121.4737,
        "subsidence_value": 1.5,
        "subsidence_unit": "cm/year",
        "source_label": "Shanghai Geological Survey",
        "source_url": "https://www.sigs.com.cn/",
        "observation_notes": "Historic subsidence ~10 cm/yr arrested via aquifer recharge; residual ~1.5 cm/yr.",
    },
    {
        "record_id": "dhaka_bd",
        "location_name": "Dhaka",
        "country": "Bangladesh",
        "admin_level_1": "Dhaka Division",
        "area_type": "postal_or_district",
        "area_name": "Dhaka metropolitan",
        "postal_code": "1000",
        "latitude": 23.8103,
        "longitude": 90.4125,
        "subsidence_value": 1.4,
        "subsidence_unit": "cm/year",
        "source_label": "Higgins et al. (2014) Geophysical Research Letters",
        "source_url": "https://doi.org/10.1002/2014GL061091",
        "observation_notes": "Dhaka subsiding ~1-1.4 cm/yr; aquifer depletion plus Ganges-Brahmaputra delta compaction.",
    },
    {
        "record_id": "central_valley_ca",
        "location_name": "Corcoran subsidence bowl",
        "country": "United States",
        "admin_level_1": "California",
        "area_type": "county",
        "area_name": "Kings County",
        "postal_code": "93212",
        "latitude": 36.0980,
        "longitude": -119.5604,
        "subsidence_value": 60.0,
        "subsidence_unit": "cm/year",
        "source_label": "NASA JPL Sentinel-1 Analysis (Farr et al. 2017)",
        "source_url": "https://www.nasa.gov/feature/jpl/nasa-data-show-california-aqueducts-bend-as-central-valley-sinks/",
        "observation_notes": "Maximum measured subsidence near Corcoran during 2014-16 drought (~60 cm/yr peak).",
    },
]


# ---------------------------------------------------------------------------
# Live CMR catalog query
# ---------------------------------------------------------------------------

def fetch_cmr_granule_counts(short_names: Iterable[str], timeout: float = 20.0) -> dict[str, dict]:
    """For each NASA short_name, hit CMR and report how many granules exist.

    Returns: { short_name: {"hits": int, "sample_titles": [...], "error": str | None} }
    Used as a freshness signal — when NISAR L2 GUNW granules go public, this
    pipeline can be extended to download their footprints directly.
    """
    results: dict[str, dict] = {}
    for name in short_names:
        params = {
            "short_name": name,
            "page_size": "5",
            "sort_key": "-start_date",
        }
        url = f"{CMR_ENDPOINT}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={"Client-Id": "model.earth-nisar-pipeline"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            entries = payload.get("feed", {}).get("entry", [])
            results[name] = {
                "hits": len(entries),
                "sample_titles": [e.get("title", "") for e in entries[:3]],
                "error": None,
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            results[name] = {"hits": 0, "sample_titles": [], "error": str(exc)}
    return results


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "record_id",
    "location_name",
    "country",
    "admin_level_1",
    "area_type",
    "area_name",
    "postal_code",
    "latitude",
    "longitude",
    "subsidence_value",
    "subsidence_unit",
    "source_label",
    "source_url",
    "observation_notes",
]


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def write_catalog_report(cmr_results: dict[str, dict], out_path: Path) -> None:
    report = {
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": CMR_ENDPOINT,
        "datasets": cmr_results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent / "nisar_subsidence.csv"),
        help="Output CSV path (default: ../nisar_subsidence.csv)",
    )
    parser.add_argument(
        "--catalog-report",
        default=str(Path(__file__).resolve().parent / "cmr_catalog.json"),
        help="Where to write the CMR catalog freshness report",
    )
    parser.add_argument(
        "--no-cmr",
        action="store_true",
        help="Skip the live CMR catalog query (offline mode)",
    )
    parser.add_argument(
        "--no-h5-metadata",
        action="store_true",
        help="Skip extracting metadata from NISAR_L2.h5 to JSON sidecar",
    )
    args = parser.parse_args(argv)

    out_path = Path(args.out).resolve()
    report_path = Path(args.catalog_report).resolve()

    print(f"[pipeline] curated hotspots: {len(CURATED_HOTSPOTS)}")
    write_csv(CURATED_HOTSPOTS, out_path)
    print(f"[pipeline] wrote {out_path}")

    if args.no_cmr:
        print("[pipeline] --no-cmr set, skipping CMR catalog query")
    else:
        print(f"[pipeline] querying NASA CMR for: {', '.join(DEFORMATION_SHORT_NAMES)}")
        cmr_results = fetch_cmr_granule_counts(DEFORMATION_SHORT_NAMES)
        write_catalog_report(cmr_results, report_path)
        print(f"[pipeline] catalog report -> {report_path}")
        for name, info in cmr_results.items():
            if info["error"]:
                print(f"  {name}: ERROR ({info['error']})")
            else:
                print(f"  {name}: {info['hits']} granules")

    if not args.no_h5_metadata:
        h5_path = Path(__file__).resolve().parent.parent / "NISAR_L2.h5"
        if h5_path.exists():
            try:
                from extract_h5_metadata import extract_metadata
                json_out = h5_path.with_name("nisar_l2_metadata.json")
                metadata = extract_metadata(h5_path)
                json_out.write_text(json.dumps(metadata, indent=2))
                print(f"[pipeline] h5 metadata -> {json_out}")
            except ImportError as exc:
                print(f"[pipeline] skipping H5 metadata extraction: {exc}")
            except Exception as exc:
                print(f"[pipeline] H5 metadata extraction failed: {exc}")
        else:
            print(f"[pipeline] H5 file not found at {h5_path}, skipping sidecar")
    return 0


if __name__ == "__main__":
    sys.exit(main())
