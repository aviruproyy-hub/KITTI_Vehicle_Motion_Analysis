
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer


# --------------------------------------------------
# 1. AUTO-DISCOVER ALL KITTI SEQUENCES
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "raw_data" / "2011_09_26"

OUTPUT_DIR = PROJECT_DIR / "processed_data"
PLOT_DIR = PROJECT_DIR / "visualizations"

OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# Find every sequence folder that contains oxts data.
sequence_dirs = sorted([
    d for d in RAW_DIR.iterdir()
    if d.is_dir()
    and (d / "oxts" / "data").exists()
    and d.name.endswith("_sync")
])

if not sequence_dirs:
    raise FileNotFoundError(
        f"No KITTI sequences found in: {RAW_DIR}"
    )

print(f"Found {len(sequence_dirs)} sequences:")
for d in sequence_dirs:
    print(f"  - {d.name}")
print()


# --------------------------------------------------
# 2. FUNCTION: PROCESS ONE SEQUENCE
# --------------------------------------------------

def process_sequence(seq_dir):
    """Process a single KITTI sequence and return a DataFrame."""

    seq_name = seq_dir.name
    oxts_dir = seq_dir / "oxts" / "data"
    timestamp_file = seq_dir / "oxts" / "timestamps.txt"

    # Read OXTS files
    files = sorted(oxts_dir.glob("*.txt"))

    if not files:
        print(f"  WARNING: No OXTS files in {seq_name}, skipping.")
        return None

    measurements = []
    for file in files:
        values = np.loadtxt(file)
        if len(values) != 30:
            print(
                f"  WARNING: {file.name} has {len(values)} values "
                f"(expected 30), skipping sequence."
            )
            return None
        measurements.append(values)

    data = np.asarray(measurements)

    # Read timestamps
    if not timestamp_file.exists():
        print(f"  WARNING: No timestamps.txt in {seq_name}, skipping.")
        return None

    timestamp_lines = timestamp_file.read_text().splitlines()
    timestamps = pd.to_datetime(timestamp_lines)

    if len(timestamps) != len(data):
        print(
            f"  WARNING: Timestamp/OXTS mismatch in {seq_name}, skipping."
        )
        return None

    elapsed_seconds = (
        timestamps - timestamps[0]
    ).total_seconds()

    # Extract vehicle data
    latitude = data[:, 0]
    longitude = data[:, 1]
    altitude = data[:, 2]

    roll = data[:, 3]
    pitch = data[:, 4]
    yaw = data[:, 5]

    velocity_north = data[:, 6]
    velocity_east = data[:, 7]
    forward_speed = data[:, 8]

    forward_acceleration = data[:, 14]
    lateral_acceleration = data[:, 15]

    speed_mps = np.hypot(velocity_north, velocity_east)
    speed_kmh = speed_mps * 3.6

    # Convert GPS to local X, Y coordinates
    utm_zone = int(
        (np.median(longitude) + 180) // 6
    ) + 1
    epsg_code = 32600 + utm_zone

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg_code}",
        always_xy=True
    )

    easting, northing = transformer.transform(
        longitude, latitude
    )

    x = easting - easting[0]
    y = northing - northing[0]

    # Build DataFrame for this sequence
    df = pd.DataFrame({
        "sequence": seq_name,
        "frame_index": np.arange(len(data)),
        "timestamp": timestamps.astype(str),
        "elapsed_time_s": elapsed_seconds,

        "latitude": latitude,
        "longitude": longitude,
        "altitude_m": altitude,

        "x_m": x,
        "y_m": y,

        "roll_rad": roll,
        "pitch_rad": pitch,
        "heading_yaw_rad": yaw,
        "heading_deg": np.rad2deg(yaw),

        "speed_mps": speed_mps,
        "speed_kmh": speed_kmh,
        "forward_speed_mps": forward_speed,

        "forward_acceleration_mps2": forward_acceleration,
        "lateral_acceleration_mps2": lateral_acceleration,
    })

    print(f"  {seq_name}: {len(df)} frames, "
          f"{round(elapsed_seconds[-1], 1)}s duration")

    return df


# --------------------------------------------------
# 3. PROCESS ALL SEQUENCES
# --------------------------------------------------

all_frames = []

for seq_dir in sequence_dirs:
    df_seq = process_sequence(seq_dir)
    if df_seq is not None:
        all_frames.append(df_seq)

if not all_frames:
    raise RuntimeError("No sequences were processed successfully.")

df = pd.concat(all_frames, ignore_index=True)

print(f"\nTotal frames across all sequences: {len(df)}")


# --------------------------------------------------
# 4. SAVE COMBINED CSV
# --------------------------------------------------

csv_path = OUTPUT_DIR / "kitti_vehicle_motion.csv"
df.to_csv(csv_path, index=False)
print(f"CSV saved: {csv_path}")


# --------------------------------------------------
# 5. GRAPH 1: ALL TRAJECTORIES (one colour per sequence)
# --------------------------------------------------

plt.figure(figsize=(11, 8))

for seq_name, group in df.groupby("sequence"):
    plt.plot(
        group["x_m"].values,
        group["y_m"].values,
        label=seq_name.replace("2011_09_26_drive_", "").replace("_sync", ""),
        linewidth=1.2
    )

plt.xlabel("East displacement (m)")
plt.ylabel("North displacement (m)")
plt.title("KITTI Vehicle Trajectories (All Sequences)")
plt.axis("equal")
plt.grid(True)
plt.legend(fontsize=7, loc="best")
plt.tight_layout()
plt.savefig(PLOT_DIR / "trajectory_xy.png", dpi=200)
plt.close()


# --------------------------------------------------
# 6. GRAPH 2: SPEED OVER TIME (per sequence)
# --------------------------------------------------

plt.figure(figsize=(11, 6))

for seq_name, group in df.groupby("sequence"):
    plt.plot(
        group["elapsed_time_s"].values,
        group["speed_kmh"].values,
        label=seq_name.replace("2011_09_26_drive_", "").replace("_sync", ""),
        linewidth=0.9,
        alpha=0.8
    )

plt.xlabel("Time within sequence (seconds)")
plt.ylabel("Speed (km/h)")
plt.title("Vehicle Speed Over Time (All Sequences)")
plt.grid(True)
plt.legend(fontsize=7, loc="best")
plt.tight_layout()
plt.savefig(PLOT_DIR / "speed_over_time.png", dpi=200)
plt.close()


# --------------------------------------------------
# 7. GRAPH 3: HEADING OVER TIME (per sequence)
# --------------------------------------------------

plt.figure(figsize=(11, 6))

for seq_name, group in df.groupby("sequence"):
    yaw_vals = group["heading_yaw_rad"].values
    heading_unwrapped = np.unwrap(yaw_vals)
    heading_degrees = np.rad2deg(heading_unwrapped)

    plt.plot(
        group["elapsed_time_s"].values,
        heading_degrees,
        label=seq_name.replace("2011_09_26_drive_", "").replace("_sync", ""),
        linewidth=0.9,
        alpha=0.8
    )

plt.xlabel("Time within sequence (seconds)")
plt.ylabel("Heading (degrees)")
plt.title("Vehicle Heading Over Time (All Sequences)")
plt.grid(True)
plt.legend(fontsize=7, loc="best")
plt.tight_layout()
plt.savefig(PLOT_DIR / "heading_over_time.png", dpi=200)
plt.close()


# --------------------------------------------------
# 8. PRINT SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

print(f"Sequences processed: {df['sequence'].nunique()}")
print(f"Total frames:        {len(df)}")
print(f"Average speed:       {round(df['speed_kmh'].mean(), 2)} km/h")
print(f"Maximum speed:       {round(df['speed_kmh'].max(), 2)} km/h")

print("\nPer-sequence breakdown:")
for seq_name, group in df.groupby("sequence"):
    short = seq_name.replace("2011_09_26_drive_", "").replace("_sync", "")
    print(
        f"  {short}: {len(group)} frames, "
        f"avg {round(group['speed_kmh'].mean(), 1)} km/h, "
        f"max {round(group['speed_kmh'].max(), 1)} km/h"
    )

print("\nProcessing completed successfully!")
