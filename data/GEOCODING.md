# Location Geocoding Documentation

Per organizer approval (2026-07-02), `name_location` strings from the competition
metadata were converted to geographic coordinates using a free, publicly available,
reproducible geocoding service. This document records the exact procedure so the
result can be reproduced at verification time.

## Source

- **Service:** Nominatim (OpenStreetMap) — https://nominatim.openstreetmap.org/search
- **License:** Data © OpenStreetMap contributors, ODbL. Free for use with the
  [Nominatim usage policy](https://operations.osmfoundation.org/policies/nominatim/)
  (max 1 request/second, custom User-Agent).
- **Retrieval date:** 2026-07-03
- **CRS:** EPSG:4326 (WGS84 latitude/longitude), as returned by Nominatim.

## Procedure

1. For each of the 38 unique `name_location` values (20 train + 18 evaluation),
   underscores were replaced with spaces and the string was sent to the Nominatim
   search API (`format=json`, `limit=1`), at 1 request/second with a custom
   User-Agent.
2. The top result's `lat`/`lon` were taken as the location coordinates.
3. Six names were ambiguous or matched an unintended region with the plain query.
   For these, a qualified query (adding country/state context derived from the
   competition's satellite coverage: Himawari=Asia-Oceania, GOES=Americas,
   Meteosat=Europe/Africa) was used instead. The qualified queries are listed
   below and are part of the reproducible procedure.

## Qualified queries

| name_location | query sent to Nominatim | reason |
|---|---|---|
| atlantic_coast | `Atlantic Coast, North Carolina, United States` | plain query matched an unrelated place; GOES coverage implies US Atlantic coast |
| upper_midwest | `Upper Midwest, United States` | plain query is not a gazetteer entry by itself |
| central_philippines | `Visayas, Philippines` | "Central Philippines" is a descriptive region; Visayas is its standard name |
| central_vietnam | `Mien Trung, Vietnam` | "Central Vietnam" is a descriptive region; Miền Trung is its standard name |
| quang_nam | `Quang Nam Province, Vietnam` | plain query matched a street-level object |
| northeast_malaysia | `Kelantan, Malaysia` | "Northeast Malaysia" is descriptive; Kelantan is the northeastern state of Peninsular Malaysia |

All other 32 names used the plain query (underscores → spaces) unmodified.

## Output files

- `location_coordinates_geocoded.csv` — `name_location,latitude,longitude`
  (the file consumed by training code)
- `location_coordinates_geocoded_full.csv` — additionally records the exact
  `query` sent and the `display_name` returned by Nominatim for auditability.

## Reproduction

```python
import requests, time

def geocode(query):
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": 1},
        headers={"User-Agent": "solafune-nowcast-research/1.0"},
        timeout=30,
    )
    r.raise_for_status()
    hit = r.json()[0]
    time.sleep(1.0)  # Nominatim usage policy
    return float(hit["lat"]), float(hit["lon"])
```

Applying this function to the queries in `location_coordinates_geocoded_full.csv`
(`query` column) reproduces the `latitude`/`longitude` columns.
