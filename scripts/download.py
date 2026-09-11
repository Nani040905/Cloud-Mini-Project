"""
scripts/download.py

Stream-filter pipeline for Google Cluster Trace v2 (2011).
Downloads machine_events to select 300 candidate machines (seed=42),
then downloads task_usage parts one by one, filters rows to candidates,
saves filtered parquet chunks, and immediately deletes raw data.
"""

import os
import sys
import json
import argparse
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Configuration (Locked in PROJECT_SPEC.md) ---
SEED              = 42
N_CANDIDATES      = 300
TOTAL_PARTS       = 500
BASE_URL          = "https://storage.googleapis.com/clusterdata-2011-2"

PROJECT_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_EVENTS_DIR    = os.path.join(PROJECT_ROOT, "data", "raw", "google", "machine_events")
RAW_USAGE_DIR     = os.path.join(PROJECT_ROOT, "data", "raw", "google", "task_usage")
PROCESSED_DIR     = os.path.join(PROJECT_ROOT, "data", "processed", "google", "filtered_parts")
CANDIDATES_FILE   = os.path.join(PROJECT_ROOT, "data", "candidate_machines.json")

COLS_USAGE        = [0, 1, 4, 5, 6, 11]
COL_NAMES_USAGE   = ['start_time', 'end_time', 'machine_id', 'cpu_rate', 'canonical_memory_usage', 'disk_io_time']


import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.compute as pc
import pyarrow.parquet as pq

CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB chunk size (maximizes socket throughput, zero throttling)

USAGE_DTYPES = {
    'start_time': 'float64',
    'end_time': 'float64',
    'machine_id': 'float64',
    'cpu_rate': 'float32',
    'canonical_memory_usage': 'float32',
    'disk_io_time': 'float32'
}


def create_session() -> requests.Session:
    """Create a requests Session with HTTP keep-alive and connection pooling."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=retries)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


GLOBAL_SESSION = create_session()


def download_file(url: str, dest_path: str, desc: str = "", max_retries: int = 5, session: requests.Session = GLOBAL_SESSION):
    """Download a file with streaming, progress bar, automatic resume, and retry logic without artificial speed limits."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    head_resp = session.head(url, timeout=30)
    head_resp.raise_for_status()
    total_size = int(head_resp.headers.get('content-length', 0))

    if os.path.exists(dest_path) and os.path.getsize(dest_path) == total_size and total_size > 0:
        return

    temp_path = dest_path + ".downloading"
    for attempt in range(1, max_retries + 1):
        try:
            downloaded_bytes = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            headers = {}
            mode = 'wb'
            if 0 < downloaded_bytes < total_size:
                headers['Range'] = f"bytes={downloaded_bytes}-"
                mode = 'ab'
            elif downloaded_bytes > total_size:
                downloaded_bytes = 0
                mode = 'wb'

            resp = session.get(url, headers=headers, stream=True, timeout=(15, 60))
            if resp.status_code not in (200, 206):
                resp = session.get(url, stream=True, timeout=(15, 60))
                resp.raise_for_status()
                downloaded_bytes = 0
                mode = 'wb'

            with open(temp_path, mode) as f, tqdm(
                desc=desc or os.path.basename(dest_path),
                total=total_size,
                initial=downloaded_bytes,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                leave=False
            ) as bar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            final_size = os.path.getsize(temp_path)
            if total_size > 0 and final_size != total_size:
                raise IOError(f"Incomplete download: {final_size}/{total_size} bytes")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.replace(temp_path, dest_path)
            return

        except Exception as e:
            print(f"\n[Warning] Attempt {attempt}/{max_retries} failed for {desc}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise


def step1_get_candidates(force: bool = False) -> set:
    """Download machine_events and select 300 candidate machines."""
    events_file = os.path.join(RAW_EVENTS_DIR, "part-00000-of-00001.csv.gz")
    if not os.path.exists(events_file):
        url = f"{BASE_URL}/machine_events/part-00000-of-00001.csv.gz"
        print("Downloading machine_events...")
        download_file(url, events_file, desc="machine_events")
    else:
        print(f"Found existing machine_events at: {events_file}")

    if os.path.exists(CANDIDATES_FILE) and not force:
        with open(CANDIDATES_FILE, 'r') as f:
            candidates = set(json.load(f))
        print(f"Loaded {len(candidates)} existing candidates from {CANDIDATES_FILE}")
        return candidates

    print("Extracting machine IDs from machine_events...")
    machine_events = pd.read_csv(
        events_file,
        header=None,
        names=['time', 'machine_id', 'event', 'platform', 'cpus', 'memory']
    )
    all_machine_ids = machine_events['machine_id'].dropna().unique()
    print(f"Found {len(all_machine_ids)} unique machine IDs in cluster trace.")

    rng = np.random.default_rng(SEED)
    chosen_candidates = sorted(rng.choice(all_machine_ids, size=N_CANDIDATES, replace=False).tolist())

    with open(CANDIDATES_FILE, 'w') as f:
        json.dump(chosen_candidates, f, indent=2)
    print(f"Selected {len(chosen_candidates)} candidate machines (seed={SEED}) saved to {CANDIDATES_FILE}")

    return set(chosen_candidates)


def process_single_part(part_idx: int, cand_arr: pa.Array, force: bool = False):
    """Download, filter using native PyArrow C++, save parquet, and clean up raw csv.gz immediately."""
    parquet_name = f"filtered_{part_idx:05d}.parquet"
    parquet_path = os.path.join(PROCESSED_DIR, parquet_name)

    if os.path.exists(parquet_path) and not force:
        return part_idx, 0, True

    part_name    = f"part-{part_idx:05d}-of-00500.csv.gz"
    url          = f"{BASE_URL}/task_usage/{part_name}"
    raw_local    = os.path.join(RAW_USAGE_DIR, part_name)

    try:
        t_dl_start = time.time()
        download_file(url, raw_local, desc=f"Part {part_idx:03d}")
        dl_time = time.time() - t_dl_start
        file_size_mb = os.path.getsize(raw_local) / (1024 * 1024)
        dl_speed = file_size_mb / max(dl_time, 0.001)

        t_filter_start = time.time()
        # Native PyArrow multithreaded C++ parser: auto-detect columns, load only needed cols, zero pandas overhead
        read_options = pv.ReadOptions(autogenerate_column_names=True)
        convert_options = pv.ConvertOptions(
            include_columns=['f0', 'f1', 'f4', 'f5', 'f6', 'f11'],
            column_types={
                'f0': pa.float64(),
                'f1': pa.float64(),
                'f4': pa.float64(),
                'f5': pa.float32(),
                'f6': pa.float32(),
                'f11': pa.float32()
            }
        )
        table = pv.read_csv(raw_local, read_options=read_options, convert_options=convert_options)
        table = table.rename_columns(COL_NAMES_USAGE)
        mask = pc.is_in(table['machine_id'], cand_arr)
        filtered_table = table.filter(mask)

        # Atomic write to temporary file then rename to prevent partial parquets
        temp_parquet = parquet_path + ".tmp"
        pq.write_table(filtered_table, temp_parquet, compression='snappy')
        if os.path.exists(parquet_path):
            os.remove(parquet_path)
        os.replace(temp_parquet, parquet_path)

        t_filter = time.time() - t_filter_start
        rows_kept = len(filtered_table)
        print(f"[{part_idx + 1:03d}/{TOTAL_PARTS}] DL {file_size_mb:.1f}MB ({dl_speed:.2f}MB/s) | Converted {len(table):,} -> {rows_kept:,} rows in {t_filter:.2f}s -> {parquet_name}")
        return part_idx, rows_kept, False

    finally:
        if os.path.exists(raw_local):
            try:
                os.remove(raw_local)
            except OSError:
                pass


def step2_stream_filter(candidates: set, start_part: int = 0, num_parts: int = TOTAL_PARTS,
                        force: bool = False, workers: int = 6):
    """
    Stream each task_usage part, filter rows to candidates, save parquet,
    and delete raw csv.gz immediately to conserve disk space.
    Uses ThreadPoolExecutor for 6 parallel downloads and sub-second PyArrow conversions.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RAW_USAGE_DIR, exist_ok=True)

    # Pre-build PyArrow candidate array once to avoid repetitive allocations
    cand_arr = pa.array(list(candidates), type=pa.float64())

    end_part = min(TOTAL_PARTS, start_part + num_parts)
    print(f"\n--- Streaming parts {start_part} to {end_part - 1} (total {end_part - start_part} parts, {workers} parallel workers) ---")

    # Collect part indices that actually need processing
    pending_parts = []
    for part_idx in range(start_part, end_part):
        parquet_path = os.path.join(PROCESSED_DIR, f"filtered_{part_idx:05d}.parquet")
        if not (os.path.exists(parquet_path) and not force):
            pending_parts.append(part_idx)
        else:
            print(f"[{part_idx + 1:03d}/{TOTAL_PARTS}] Part part-{part_idx:05d}-of-00500.csv.gz already processed -> skipping.")

    if not pending_parts:
        print("All requested parts are already processed.")
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_part, p_idx, cand_arr, force): p_idx for p_idx in pending_parts}
        for future in as_completed(futures):
            p_idx = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[Error] Failed processing part {p_idx}: {e}")
                raise


def main():
    parser = argparse.ArgumentParser(description="Download and stream-filter Google Cluster Trace v2")
    parser.add_argument("--start", type=int, default=0, help="Starting part index (0-499)")
    parser.add_argument("--count", type=int, default=TOTAL_PARTS, help="Number of parts to process")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel download & filter workers (default 6)")
    parser.add_argument("--test", action="store_true", help="Run test mode (process 3 parts only)")
    parser.add_argument("--force", action="store_true", help="Force re-download and re-filter even if files exist")
    args = parser.parse_args()

    candidates = step1_get_candidates(force=args.force)

    if args.test:
        print("\n*** RUNNING IN TEST MODE (3 PARTS ONLY) ***")
        step2_stream_filter(candidates, start_part=0, num_parts=3, force=args.force, workers=args.workers)
    else:
        step2_stream_filter(candidates, start_part=args.start, num_parts=args.count, force=args.force, workers=args.workers)

    print("\nProcessing complete.")


if __name__ == "__main__":
    main()
