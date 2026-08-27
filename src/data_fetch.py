"""All external API calls: USGS groundwater levels (NGWMN) and NASA POWER climate data."""

import time
import requests
import pandas as pd

from src import config


def _extract_lat_lon(sites: pd.DataFrame) -> pd.DataFrame:
    """Pull lat/lon out of whatever form the API returned them in.

    Without geopandas installed, this package returns coordinates as a raw
    [lon, lat] list inside a single 'geometry' column rather than separate
    'latitude'/'longitude' columns, so we unpack that here.
    """
    sites = sites.copy()
    if "geometry" in sites.columns:
        coords = sites["geometry"].apply(
            lambda g: g if isinstance(g, (list, tuple)) and len(g) >= 2 else (None, None)
        )
        sites["longitude"] = coords.apply(lambda c: c[0])
        sites["latitude"] = coords.apply(lambda c: c[1])
    return sites


def discover_wells(state: str = config.STATE, limit: int = 50) -> pd.DataFrame:
    """List candidate NGWMN wells in a state, with lat/lon, to pick a well ID from.

    Run this interactively, not as part of the automated pipeline.
    """
    from dataretrieval import ngwmn

    sites, _ = ngwmn.get_sites(state=state, limit=limit)
    if sites is None or len(sites) == 0:
        raise RuntimeError(f"No NGWMN wells found for state='{state}'.")

    sites = _extract_lat_lon(sites)

    keep_cols = [c for c in [
        "monitoring_location_id", "monitoring_location_name",
        "latitude", "longitude", "national_aquifer_code"
    ] if c in sites.columns]

    if "latitude" not in keep_cols or "longitude" not in keep_cols:
        print("Could not find latitude/longitude columns. Full column list "
              f"for inspection: {list(sites.columns)}")

    return sites[keep_cols].dropna(subset=["monitoring_location_id"]).reset_index(drop=True)

def discover_continuous_gw_wells(state: str = config.STATE, limit: int = 50) -> pd.DataFrame:
    """List wells with continuous, automated depth-to-water sensors
    (parameter code 72019), rather than sparse hand-measured field visits.

    These give daily readings instead of a handful of readings per year,
    which is what a monthly time-series model actually needs.
    """
    from dataretrieval import waterdata

    ts, _ = waterdata.get_time_series_metadata(
        parameter_code="72019", state=state, limit=limit
    )
    if ts is None or len(ts) == 0:
        raise RuntimeError(f"No continuous GWL time series found for state='{state}'.")

    ts = _extract_lat_lon(ts)
    keep_cols = [c for c in [
        "monitoring_location_id", "monitoring_location_name",
        "latitude", "longitude", "begin", "end"
    ] if c in ts.columns]
    return ts[keep_cols].reset_index(drop=True)


def get_gwl_daily_timeseries(well_id: str, start: str = config.START_DATE,
                              end: str = config.END_DATE) -> pd.DataFrame:
    """Daily depth-to-water from a continuously-monitored well (72019),
    instead of sparse field measurements. Prefer this over
    get_gwl_timeseries() whenever the well has continuous sensor data.
    """
    from dataretrieval import waterdata

    df, _ = waterdata.get_daily(
        monitoring_location_id=well_id,
        parameter_code="72019",
        time=f"{start}/{end}",
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"No continuous daily GWL data for well '{well_id}'.")

    time_col = next((c for c in ["time", "datetime", "date"] if c in df.columns), None)
    value_col = "value" if "value" in df.columns else None
    if time_col is None or value_col is None:
        raise RuntimeError(f"Unrecognized daily-GWL columns for well '{well_id}': {list(df.columns)}")

    out = df[[time_col, value_col]].rename(columns={time_col: "date", value_col: "gwl_ft_below_surface"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.dropna(subset=["gwl_ft_below_surface"]).sort_values("date").reset_index(drop=True)

def get_gwl_timeseries(well_id: str, start: str = config.START_DATE,
                        end: str = config.END_DATE) -> pd.DataFrame:
    """Pull the water-level time series for one well from NGWMN.

    Returns columns: date, gwl_ft_below_surface (depth to water in feet;
    higher = deeper/drier).
    """
    from dataretrieval import ngwmn

    df, _ = ngwmn.get_water_level(monitoring_location_id=well_id, datetime=[start, end])
    if df is None or len(df) == 0:
        raise RuntimeError(
            f"No water-level records for well '{well_id}' between {start} and {end}. "
            "Try a different well or widen the date range."
        )

    # Column names have shifted between package versions before, so match
    # the first plausible name rather than assuming one exact string.
    time_col = next((c for c in ["sample_time", "datetime", "time"] if c in df.columns), None)
    value_col = next((c for c in [
        "water_depth_below_land_surface_ft", "depth_to_water_ft"
    ] if c in df.columns), None)

    if time_col is None or value_col is None:
        raise RuntimeError(
            f"Unrecognized NGWMN response columns for well '{well_id}': {list(df.columns)}"
        )

    out = df[[time_col, value_col]].rename(
        columns={time_col: "date", value_col: "gwl_ft_below_surface"}
    )
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.dropna(subset=["gwl_ft_below_surface"]).sort_values("date").reset_index(drop=True)


def get_climate_data(lat: float, lon: float, start: str = config.START_DATE,
                      end: str = config.END_DATE, max_retries: int = 3) -> pd.DataFrame:
    """Daily rainfall (mm) and mean temperature (C) for a lat/lon from NASA POWER."""
    params = {
        "parameters": config.NASA_POWER_PARAMETERS,
        "community": config.NASA_POWER_COMMUNITY,
        "longitude": lon,
        "latitude": lat,
        "start": pd.Timestamp(start).strftime("%Y%m%d"),
        "end": pd.Timestamp(end).strftime("%Y%m%d"),
        "format": "JSON",
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(config.NASA_POWER_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            break
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2 * attempt)
    else:
        raise RuntimeError(f"NASA POWER request failed after {max_retries} attempts: {last_error}")

    daily = payload["properties"]["parameter"]
    rain = daily.get("PRECTOTCORR", {})
    temp = daily.get("T2M", {})
    dates = sorted(set(rain) | set(temp))

    df = pd.DataFrame({
        "date": pd.to_datetime(dates, format="%Y%m%d"),
        "rainfall_mm": [rain.get(d) for d in dates],
        "temp_c": [temp.get(d) for d in dates],
    })
    # NASA POWER uses -999 as a missing-value sentinel.
    df.loc[df["rainfall_mm"] <= -900, "rainfall_mm"] = None
    df.loc[df["temp_c"] <= -900, "temp_c"] = None
    return df.sort_values("date").reset_index(drop=True)


def load_backup_dataset(csv_path: str) -> pd.DataFrame:
    """Load a manually-downloaded backup dataset (e.g. Jasechko et al. 2024
    from HydroShare) if a chosen well/region has too little USGS history."""
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    print(f"Discovering wells in {config.STATE} ...")
    print(discover_wells(limit=10))
    print(f"\nContinuously-monitored wells (denser data) in {config.STATE}:")
    print(discover_continuous_gw_wells(limit=20))