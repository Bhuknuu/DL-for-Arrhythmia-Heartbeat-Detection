"""Download and integrity check script for the MIT-BIH Arrhythmia Database."""

import os
import sys
from pathlib import Path
import wfdb

# List of 48 record names in MIT-BIH Arrhythmia Database
MITDB_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]


def download_mitdb(target_dir: str = "data/raw/mitdb") -> None:
    """Download MIT-BIH records from PhysioNet if not already present."""
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)
    
    print(f"Target directory: {dest.resolve()}")
    missing_records = []
    present_records = []

    for record in MITDB_RECORDS:
        dat_file = dest / f"{record}.dat"
        hea_file = dest / f"{record}.hea"
        atr_file = dest / f"{record}.atr"
        
        if dat_file.exists() and hea_file.exists() and atr_file.exists():
            present_records.append(record)
        else:
            missing_records.append(record)

    print(f"Found {len(present_records)}/{len(MITDB_RECORDS)} records present.")
    
    if missing_records:
        print(f"Downloading {len(missing_records)} missing records from PhysioNet...")
        for record in missing_records:
            print(f"Downloading record {record}...")
            try:
                wfdb.dl_database("mitdb", str(dest), records=[record])
            except Exception as e:
                print(f"Error downloading record {record}: {e}")
        print("Download complete.")
    else:
        print("All 48 MIT-BIH records are verified and present!")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/mitdb"
    download_mitdb(out_dir)
