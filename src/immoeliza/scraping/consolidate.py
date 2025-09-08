import pandas as pd
from pathlib import Path
from .common import logger
from ..io_paths import today_partition, raw_path

def _guess_is_kind(url: str, kind: str) -> bool:
    """Heuristic to separate apartments vs houses for legacy runs."""
    k = "apartment" if kind == "apartments" else "house"
    u = (url or "").lower()
    return f"/{k}/" in u or f"{k}?" in u or f"{k}&" in u

def consolidate_urls(kind: str) -> str:
    # modern layout: data/raw/YYYY/MM/DD/{kind}/**/*.csv
    day_dir = today_partition()
    modern_dir = day_dir / kind
    candidates = []

    if modern_dir.exists():
        candidates.extend(modern_dir.rglob("*.csv"))

    # legacy layout: data/raw/YYYY/MM/DD/<town_runid>/*.csv
    # only include files with URLs that look like the target kind
    legacy_dirs = [p for p in day_dir.iterdir() if p.is_dir() and p.name not in ("apartments", "houses")]
    for d in legacy_dirs:
        candidates.extend(d.glob("*.csv"))

    if not candidates:
        raise FileNotFoundError(f"No CSV files under {modern_dir if modern_dir.exists() else day_dir}")

    frames = []
    for p in sorted(set(candidates)):
        try:
            df = pd.read_csv(p)
            # normalise 'url' column
            url_col = next((c for c in df.columns if c.lower() in {"url","urls","link","href"}), None)
            if not url_col:
                continue
            df.rename(columns={url_col: "url"}, inplace=True)
            # if from legacy dirs (not under {kind}), filter rows by kind
            if p.is_relative_to(day_dir) and not p.is_relative_to(modern_dir):
                df = df[df["url"].astype(str).apply(lambda u: _guess_is_kind(u, kind))]
                if df.empty:
                    continue
            df["__source_file"] = str(p)
            frames.append(df)
        except Exception as e:
            logger.warning("Skipping %s: %s", p, e)

    if not frames:
        raise RuntimeError(f"No usable URL rows for kind={kind}")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["url"]).drop_duplicates(subset=["url"])

    out = raw_path(f"{kind}_urls")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("💾 Consolidated %d URLs -> %s", len(df), out)
    return str(out)
