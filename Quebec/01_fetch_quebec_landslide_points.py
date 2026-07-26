"""
01_fetch_quebec_landslide_points.py
=====================================
Downloads Quebec landslide EVENT POINTS (not polygons) from the MSP's
historical civil-security events database, filtered to "Mouvement de
terrain" (landslide/ground movement).

WHY POINTS, NOT POLYGONS -- read before using this:
Unlike Oregon's SLIDO (which has real mapped landslide scar polygons),
Quebec's genuinely open, easily-downloadable landslide data is POINT-based.
The actual polygon scar inventory Shahabi et al. (2024) used (citing
Demers et al. 2014/2017 and Poulin Leboeuf 2020) is not confirmed to be
openly downloadable -- it appears to be an inventory the authors had
direct access to via their Geological Survey of Canada co-author, not
something sitting on Données Québec with a public API.

This means your labels here will be circular buffers around a point, not
real scarp shapes. This is a REAL limitation, not a placeholder to silently
fix later -- buffered-circle labels cannot capture scarp orientation or
true extent the way SLIDO polygons did, and the model trained on this data
will show a weaker, less specific "high slope near a point" signal rather
than a true scarp-boundary signature. Treat this MVP as "does the Quebec
data pipeline work end-to-end," not "does lidar detect Quebec landslides
well" -- those are different claims.

CONFIRMED (via live query, 2026-07-10):
- Endpoint: https://geoegl.msp.gouv.qc.ca/apis/wss/historiquesc.fcgi
- Layer: vg_observation_v_autre_wmst (this is the ARCHIVE/historical layer,
  which has far more records and longer time coverage than the "current"
  msp_risc_evenements_public layer -- picked deliberately given the
  sample-size goal)
- Fields confirmed present: date_observation, code_municipalite, nom,
  coordonnee_x, coordonnee_y, urgence, certitude, type, severite, etat,
  imprecision
- type == "Mouvement de terrain" is the landslide filter value (confirmed
  present in live data, e.g. Sainte-Marie 2012, Saint-Jude 2012, several
  Gaspé/Rivière-Ouelle/La Pocatière records back to the 1990s)
- imprecision field flags location/date uncertainty per record -- check
  this before treating a point as reliable, same spirit as SLIDO's
  CONFIDENCE field

RUN THIS LOCALLY -- this domain is not on Claude's sandbox network allowlist.

Usage:
    python 01_fetch_quebec_landslide_points.py
Output:
    quebec_landslide_points.geojson
"""
import json
import requests

WFS_URL = "https://geoegl.msp.gouv.qc.ca/apis/wss/historiquesc.fcgi"
LAYER = "vg_observation_v_autre_wmst"
LANDSLIDE_TYPE = "Mouvement de terrain"


def fetch_all_events(layer=LAYER):
    """
    Fetch the full archive layer. No server-side filtering by 'type' is
    confirmed to work reliably via this WFS (not tested), so this pulls
    everything and filters client-side -- safer than guessing at a CQL
    filter syntax the service might reject.
    """
    params = {
        "service": "wfs",
        "version": "1.1.0",
        "request": "getfeature",
        "typename": layer,
        "outputformat": "geojson",
        "srsName": "epsg:4326",
    }
    print(f"Fetching {layer} (this may be a large response, archive layer)...")
    resp = requests.get(WFS_URL, params=params, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    print(f"Fetched {len(data.get('features', []))} total event records")
    return data


def filter_landslides(data, hazard_type=LANDSLIDE_TYPE):
    features = data.get("features", [])
    landslide_features = [
        f for f in features
        if f.get("properties", {}).get("type") == hazard_type
    ]
    print(f"Filtered to {len(landslide_features)} '{hazard_type}' events")

    # Summarize imprecision -- worth knowing how many records have
    # location/date uncertainty flagged before treating this as clean data
    imprecision_counts = {}
    for f in landslide_features:
        val = f.get("properties", {}).get("imprecision", "unknown")
        imprecision_counts[val] = imprecision_counts.get(val, 0) + 1
    print(f"Imprecision breakdown: {imprecision_counts}")

    # Summarize by decade -- useful sanity check on temporal spread
    from collections import Counter
    decades = Counter()
    for f in landslide_features:
        date_str = f.get("properties", {}).get("date_observation", "")
        if date_str and len(date_str) >= 4:
            year = date_str[:4]
            if year.isdigit():
                decade = f"{year[:3]}0s"
                decades[decade] += 1
    print(f"By decade: {dict(sorted(decades.items()))}")

    return landslide_features


if __name__ == "__main__":
    data = fetch_all_events()
    landslides = filter_landslides(data)

    out = {"type": "FeatureCollection", "features": landslides}
    with open("quebec_landslide_points.geojson", "w") as fh:
        json.dump(out, fh)

    print(f"\nSaved {len(landslides)} landslide point events to quebec_landslide_points.geojson")
    print("\nNEXT STEP: these are POINTS. Before rasterizing into training")
    print("masks, you need to pick a buffer radius (see 02_buffer_and_rasterize.py).")
    print("There's no principled 'correct' radius here since we don't have real")
    print("scarp extents -- pick something defensible (e.g. based on typical")
    print("Quebec sensitive-clay landslide sizes from literature) and state it")
    print("explicitly as an assumption in your methods.")
