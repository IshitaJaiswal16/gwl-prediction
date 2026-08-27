"""Central settings: well selection, date range, file paths.

Change values here rather than scattering constants across the pipeline.
"""

from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Region used for well discovery.
STATE = "California"

# Wells to actually pull data for, as (well_id, lat, lon) tuples.
# Fill this in after running `python -m src.data_fetch` and picking a well
# from the printed list.
# WELL_IDS: list[tuple[str, float, float]] = []
WELL_IDS: list[tuple[str, float, float]] = [
    ("USGS-340046117020804", 34.012917, -117.036392),
]

# NASA POWER has data from 1981 onward; USGS coverage varies by well.
START_DATE = "2000-01-01"
END_DATE = "2025-01-01"

NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_POWER_PARAMETERS = "T2M,PRECTOTCORR"  # mean temp (C), corrected precip (mm/day)
NASA_POWER_COMMUNITY = "AG"

LAG_MONTHS = [1, 3, 6]
TEST_FRACTION = 0.20
RANDOM_SEED = 42
