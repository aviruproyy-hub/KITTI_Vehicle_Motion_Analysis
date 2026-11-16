
from pathlib import Path
from ultralytics import YOLO


# --------------------------------------------------
# 1. AUTO-DISCOVER ALL KITTI SEQUENCES
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "raw_data" / "2011_09_26"
RUNS_DIR = PROJECT_DIR / "runs" / "detect"

# Find every sequence folder that has image_02/data.
sequence_dirs = sorted([
    d for d in RAW_DIR.iterdir()
    if d.is_dir()
    and (d / "image_02" / "data").exists()
    and d.name.endswith("_sync")
])

if not sequence_dirs:
    raise FileNotFoundError(
        f"No KITTI sequences with image_02 found in: {RAW_DIR}"
    )

print(f"Found {len(sequence_dirs)} sequences to run YOLO on:")
for d in sequence_dirs:
    print(f"  - {d.name}")
print()


# --------------------------------------------------
# 2. LOAD YOLO MODEL
# --------------------------------------------------

model = YOLO("yolo26n.pt")


# --------------------------------------------------
# 3. RUN PREDICTIONS ON EACH SEQUENCE
# --------------------------------------------------

for seq_dir in sequence_dirs:

    seq_name = seq_dir.name
    image_dir = seq_dir / "image_02" / "data"
    output_name = seq_name  # e.g. 2011_09_26_drive_0001_sync

    # Check if already processed (labels folder exists).
    existing_labels = RUNS_DIR / output_name / "labels"
    if existing_labels.exists() and any(existing_labels.glob("*.txt")):
        label_count = len(list(existing_labels.glob("*.txt")))
        print(
            f"SKIP {seq_name}: already has {label_count} "
            f"label files in {existing_labels}"
        )
        continue

    print(f"Processing {seq_name}...")

    model.predict(
        source=str(image_dir),
        imgsz=640,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(RUNS_DIR),
        name=output_name,
        exist_ok=True
    )

    print(f"  Done: {seq_name}\n")


# --------------------------------------------------
# 4. SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("YOLO PREDICTION SUMMARY")
print("=" * 50)

for seq_dir in sequence_dirs:
    seq_name = seq_dir.name
    label_dir = RUNS_DIR / seq_name / "labels"

    if label_dir.exists():
        count = len(list(label_dir.glob("*.txt")))
        print(f"  {seq_name}: {count} label files")
    else:
        print(f"  {seq_name}: NO LABELS (may have had no detections)")

print("\nAll predictions completed!")
print(f"Results saved under: {RUNS_DIR}")

