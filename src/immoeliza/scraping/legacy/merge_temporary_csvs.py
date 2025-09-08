import os
import pandas as pd
import logging
from datetime import datetime

# === Préparation répertoires et fichiers ===
run_id = datetime.now().strftime("%Y%m%d_%H%M")
log_dir = "output/merge"
csv_dir = os.path.join(log_dir, "csv")
os.makedirs(csv_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"merge_log_{run_id}.log")

# === Configuration du logger ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="w", encoding="utf-8")
    ]
)

# === Paramètres configurables ===
stop_on_duplicates = False  # ✅ Mettre sur True pour arrêter après trop de doublons
max_consecutive = 10
output_dir = "output"
merged_prefix = "merged_unique_urls"

def merge_partial_csvs() -> str:
    logging.info("🚀 Starting merge process...")
    logging.info(f"🔎 Scanning directory: {output_dir}")

    all_records = []
    found = False
    prev_urls = None
    consecutive_duplicates = 0

    # Lire les fichiers CSV triés par numéro de page
    csv_files = sorted(
        [f for f in os.listdir(output_dir) if f.startswith("partial_urls_page_") and f.endswith(".csv")],
        key=lambda x: int(x.split("_")[3])
    )

    logging.info(f"📂 Found {len(csv_files)} partial CSV files to process.")
    for file in csv_files:
        path = os.path.join(output_dir, file)
        try:
            df = pd.read_csv(path)
            current_urls = set(df["url"].dropna().astype(str))

            if prev_urls is not None and current_urls == prev_urls:
                consecutive_duplicates += 1
                logging.warning(f"⚠️ Duplicate URLs detected in: {file} ({consecutive_duplicates} in a row)")
                if stop_on_duplicates and consecutive_duplicates >= max_consecutive:
                    logging.warning(f"🛑 Stopping early: {max_consecutive} consecutive duplicate pages detected.")
                    break
            else:
                consecutive_duplicates = 0

            prev_urls = current_urls
            all_records.append(df)
            logging.info(f"📄 Loaded: {file} ({len(df)} rows)")
            found = True
        except Exception as e:
            logging.warning(f"⚠️ Skipped {file} due to error: {e}")

    if not found:
        logging.error("❌ No partial CSV files found to merge.")
        return ""

    # === Fusion et déduplication ===
    merged_df = pd.concat(all_records, ignore_index=True)
    before = len(merged_df)
    merged_df.drop_duplicates(subset=["url"], inplace=True)
    after = len(merged_df)

    # Nouveau nom avec timestamp et count
    merged_filename = f"{merged_prefix}_{run_id}_{after}.csv"
    merged_csv_path = os.path.join(csv_dir, merged_filename)
    merged_df.to_csv(merged_csv_path, index=False)

    # Logs finaux
    logging.info(f"✅ Merged CSV saved: {merged_csv_path}")
    logging.info(f"🧮 Total rows before deduplication: {before}")
    logging.info(f"✅ Total unique listings after deduplication: {after}")
    logging.info(f"🎉 Merge complete! File saved at: {merged_csv_path}")

    # === Statistiques dans un fichier texte ===
    stats_path = os.path.join(log_dir, f"merge_stats_{run_id}.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"📊 Merge Summary – {run_id}\n")
        f.write(f"----------------------------------------\n")
        f.write(f"Total files scanned            : {len(csv_files)}\n")
        f.write(f"Duplicate files (same URLs)    : {consecutive_duplicates}\n")
        f.write(f"Unique files (different URLs)  : {len(csv_files) - consecutive_duplicates}\n")
        f.write(f"Rows before deduplication      : {before}\n")
        f.write(f"Unique listings after merge    : {after}\n")
        f.write(f"Merged file                    : {merged_filename}\n")

    logging.info(f"📄 Stats saved: {stats_path}")
    return merged_csv_path

# === Point d'entrée principal ===
if __name__ == "__main__":
    try:
        path = merge_partial_csvs()
        if not path:
            logging.error("❌ Merge failed or no data found.")
    except Exception as e:
        logging.exception(f"❌ Unexpected error: {e}")
        exit(1)
