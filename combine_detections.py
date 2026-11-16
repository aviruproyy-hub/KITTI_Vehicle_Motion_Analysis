
from pathlib import Path
import pandas as pd
from collections import Counter


# -----------------------------------------
# 1. FILE LOCATIONS
# -----------------------------------------

BASE = Path(__file__).resolve().parent

CSV_PATH = BASE / "processed_data" / "kitti_vehicle_motion.csv"
RUNS_DIR = BASE / "runs" / "detect"

OUTPUT_DIR = BASE / "processed_data"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "combined_motion_detections.csv"


# -----------------------------------------
# 2. COCO CLASS ID MAPPING
# -----------------------------------------

CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light"
}

count_columns = {
    "person": "persons_detected",
    "bicycle": "bicycles_detected",
    "car": "cars_detected",
    "motorcycle": "motorcycles_detected",
    "bus": "buses_detected",
    "truck": "trucks_detected",
    "traffic_light": "traffic_lights_detected"
}


# -----------------------------------------
# 3. LOAD MOTION CSV
# -----------------------------------------

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

if "sequence" not in df.columns:
    raise ValueError(
        "CSV must contain a 'sequence' column. "
        "Re-run the updated process_kitti.py first."
    )

if "frame_index" not in df.columns:
    raise ValueError("CSV must contain a 'frame_index' column.")

print(f"Loaded motion CSV: {len(df)} frames "
      f"across {df['sequence'].nunique()} sequences")


# -----------------------------------------
# 4. INITIALISE DETECTION COLUMNS
# -----------------------------------------

for column in count_columns.values():
    df[column] = 0

df["total_detections"] = 0
df["other_detections"] = 0


# -----------------------------------------
# 5. PROCESS LABELS FOR EACH SEQUENCE
# -----------------------------------------

total_processed = 0
sequences_found = 0

for seq_name in df["sequence"].unique():

    label_dir = RUNS_DIR / seq_name / "labels"

    if not label_dir.exists():
        print(f"  WARNING: No labels for {seq_name} "
              f"(expected at {label_dir})")
        continue

    label_files = sorted(label_dir.glob("*.txt"))

    if not label_files:
        print(f"  WARNING: Labels folder empty for {seq_name}")
        continue

    sequences_found += 1
    seq_processed = 0

    for label_file in label_files:

        try:
            frame_id = int(label_file.stem)
        except ValueError:
            continue

        # Match on BOTH sequence name and frame index.
        matching_rows = df.index[
            (df["sequence"] == seq_name)
            & (df["frame_index"] == frame_id)
        ]

        if len(matching_rows) == 0:
            continue

        row_index = matching_rows[0]

        counts = Counter()
        total = 0
        other = 0

        with open(label_file, "r") as file:
            for line in file:
                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(float(parts[0]))
                total += 1

                if class_id in CLASSES:
                    counts[CLASSES[class_id]] += 1
                else:
                    other += 1

        df.at[row_index, "total_detections"] = total
        df.at[row_index, "other_detections"] = other

        for class_name, column in count_columns.items():
            df.at[row_index, column] = counts[class_name]

        seq_processed += 1

    total_processed += seq_processed
    print(f"  {seq_name}: {seq_processed} label files matched")


# -----------------------------------------
# 6. SAVE COMBINED DATASET
# -----------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 50)
print("COMBINATION COMPLETED")
print("=" * 50)

print(f"Total KITTI frames:         {len(df)}")
print(f"Sequences with labels:      {sequences_found}")
print(f"Label files matched:        {total_processed}")
print(f"Output: {OUTPUT_PATH}")

print("\nDETECTION SUMMARY")

for column in count_columns.values():
    print(f"  {column}: {int(df[column].sum())}")

print(f"  Total detections: {int(df['total_detections'].sum())}")
