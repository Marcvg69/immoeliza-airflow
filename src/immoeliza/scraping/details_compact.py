from pathlib import Path
import pandas as pd
from .common import logger
from ..io_paths import today_partition, raw_path

def compact(kind: str) -> str:
    day_dir = today_partition() / kind
    if not day_dir.exists():
        raise FileNotFoundError(f"No folder for kind '{kind}' at {day_dir}")

    csvs = []
    for p in day_dir.rglob("*.csv"):
        name = p.name.lower()
        if name.startswith("partial_urls_") or name.startswith("urls_") or name.startswith("stats_"):
            continue  # skip URL CSVs and stats
        csvs.append(p)

    if not csvs:
        raise FileNotFoundError(f"No details CSVs found under {day_dir}")

    frames = []
    for p in sorted(csvs):
        try:
            df = pd.read_csv(p)
            # ensure url column exists
            url_col = next((c for c in df.columns if c.lower() == "url"), None)
            if not url_col:
                continue
            frames.append(df)
        except Exception as e:
            logger.warning("Skipping %s: %s", p, e)

    if not frames:
        raise RuntimeError("No readable details CSVs.")

    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    out = raw_path(f"{kind}_details")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("💾 Compacted %d detail rows -> %s", len(df), out)
    return str(out)
