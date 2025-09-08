from pathlib import Path
from datetime import datetime
from .config import settings

def today_partition():
    now = datetime.utcnow()
    return Path(settings.data_root) / "raw" / f"{now:%Y/%m/%d}"

def raw_path(kind: str) -> Path:
    return today_partition() / f"{kind}.parquet"

def processed_path(kind: str) -> Path:
    return Path(settings.data_root) / "processed" / f"{kind}.parquet"
