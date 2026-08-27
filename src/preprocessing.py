"""Resample raw GWL + climate data to a common monthly grid and clean up gaps."""

import pandas as pd


def resample_monthly(gwl_df: pd.DataFrame, climate_df: pd.DataFrame) -> pd.DataFrame:
    """Resample both series to monthly and merge on date.

    GWL readings are irregular (a well might be measured every few weeks),
    so monthly is the common grid both series can share. Rainfall is
    summed (it's a flux); GWL and temperature are averaged.
    """
    gwl_monthly = (
        gwl_df.set_index("date")["gwl_ft_below_surface"]
        .resample("MS")
        .mean()
        .rename("gwl")
    )
    climate_monthly = (
        climate_df.set_index("date")
        .resample("MS")
        .agg({"rainfall_mm": "sum", "temp_c": "mean"})
    )
    merged = pd.concat([gwl_monthly, climate_monthly], axis=1)
    merged.index.name = "date"
    return merged.reset_index()


def handle_missing(df: pd.DataFrame, max_consecutive_gap: int = 3) -> pd.DataFrame:
    """Linearly interpolate short gaps; drop rows where a gap is too long to
    interpolate safely (groundwater moves slowly, so a few months of
    interpolation is reasonable, but not a year-long blackout)."""
    df = df.sort_values("date").reset_index(drop=True)

    for col in ["gwl", "rainfall_mm", "temp_c"]:
        is_na = df[col].isna()
        run_id = (is_na != is_na.shift()).cumsum()
        gap_len = is_na.groupby(run_id).transform("size")
        fillable = is_na & (gap_len <= max_consecutive_gap)

        interpolated = df[col].interpolate(method="linear", limit_direction="both")
        df.loc[fillable, col] = interpolated[fillable]

    before = len(df)
    df = df.dropna(subset=["gwl", "rainfall_mm", "temp_c"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with gaps longer than "
              f"{max_consecutive_gap} months.")
    return df
