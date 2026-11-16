"""
Prepare AV Control Dataset
Phase 2 & Phase 3: Feature Engineering, Control Labels, and Train/Val/Test Split.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. PATHS AND CONSTANTS
# --------------------------------------------------

BASE = Path(__file__).resolve().parent
INPUT_CSV = BASE / "processed_data" / "combined_motion_detections.csv"
OUTPUT_DIR = BASE / "processed_data"
PLOT_DIR = BASE / "visualizations"

OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# Vehicle parameters (Volkswagen Passat B6 station wagon used in KITTI)
WHEELBASE_M = 2.71  # meters
MAX_STEERING_DEG = 35.0  # physical road wheel lock limit
MAX_STEERING_RAD = np.deg2rad(MAX_STEERING_DEG)

MAX_ACCEL_MPS2 = 3.0  # reference acceleration for throttle = 1.0 (m/s^2)
MAX_BRAKE_MPS2 = 6.0  # reference deceleration for brake = 1.0 (m/s^2)

# Sequence-based Train / Val / Test Assignment
# Ensures balanced representation of City, Residential, and Road driving
SPLIT_MAPPING = {
    # Train (~70.6%): City (0009, 0011), Residential (0019, 0022), Road (0028)
    "2011_09_26_drive_0009_sync": "train",
    "2011_09_26_drive_0011_sync": "train",
    "2011_09_26_drive_0019_sync": "train",
    "2011_09_26_drive_0022_sync": "train",
    "2011_09_26_drive_0028_sync": "train",

    # Validation (~14.4%): City (0005, 0013), Road (0027)
    "2011_09_26_drive_0005_sync": "val",
    "2011_09_26_drive_0013_sync": "val",
    "2011_09_26_drive_0027_sync": "val",

    # Test (~15.0%): City (0001, 0014), Residential (0020)
    "2011_09_26_drive_0001_sync": "test",
    "2011_09_26_drive_0014_sync": "test",
    "2011_09_26_drive_0020_sync": "test",
}


# --------------------------------------------------
# 2. LOAD COMBINED DATASET
# --------------------------------------------------

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} frames across {df['sequence'].nunique()} sequences.\n")


# --------------------------------------------------
# 3. DERIVE CONTROL LABELS (Per-Sequence)
# --------------------------------------------------

processed_sequences = []

for seq_name, group in df.groupby("sequence", sort=False):
    seq_df = group.copy().sort_values("frame_index").reset_index(drop=True)

    # Compute dt and yaw rate (rad/s) using sequence-aware numerical gradient
    time_s = seq_df["elapsed_time_s"].values
    yaw_unwrapped = np.unwrap(seq_df["heading_yaw_rad"].values)

    # Gradient handling (fallback to 0.1s KITTI nominal sampling if dt is 0)
    dt = np.gradient(time_s)
    dt = np.where(dt <= 1e-4, 0.1, dt)

    yaw_rate = np.gradient(yaw_unwrapped) / dt
    seq_df["yaw_rate_rad_s"] = yaw_rate

    # Speed (m/s)
    speed = seq_df["forward_speed_mps"].values
    # Clean up minor negative drift in forward speed
    speed = np.maximum(speed, 0.0)

    # Kinematic bicycle model: delta = atan(yaw_rate * L / v)
    # Regularized for low speeds:
    speed_threshold = 0.5  # m/s
    effective_speed = np.maximum(speed, speed_threshold)

    steering_rad = np.arctan((yaw_rate * WHEELBASE_M) / effective_speed)

    # Smooth low speed transition (if speed < 0.2 m/s, car cannot turn without moving)
    speed_factor = np.clip(speed / speed_threshold, 0.0, 1.0)
    steering_rad = steering_rad * speed_factor

    # Clip to realistic physical vehicle limits (+/- 35 degrees)
    steering_rad = np.clip(steering_rad, -MAX_STEERING_RAD, MAX_STEERING_RAD)
    steering_deg = np.rad2deg(steering_rad)

    seq_df["steering_angle_rad"] = steering_rad
    seq_df["steering_angle_deg"] = steering_deg

    # Path curvature kappa = yaw_rate / v (1/m)
    seq_df["curvature_m_inv"] = yaw_rate / effective_speed

    # Longitudinal control: Throttle & Brake
    accel = seq_df["forward_acceleration_mps2"].values
    seq_df["target_acceleration_mps2"] = accel

    # Positive acceleration -> throttle [0, 1]
    throttle = np.where(accel > 0, np.clip(accel / MAX_ACCEL_MPS2, 0.0, 1.0), 0.0)
    # Negative acceleration -> brake [0, 1]
    brake = np.where(accel < 0, np.clip(np.abs(accel) / MAX_BRAKE_MPS2, 0.0, 1.0), 0.0)

    seq_df["throttle_command"] = throttle
    seq_df["brake_command"] = brake

    # Assign split
    split = SPLIT_MAPPING.get(seq_name, "train")
    seq_df["split"] = split

    processed_sequences.append(seq_df)

final_df = pd.concat(processed_sequences, ignore_index=True)


# --------------------------------------------------
# 4. SAVE DATASETS
# --------------------------------------------------

master_path = OUTPUT_DIR / "final_av_control_dataset.csv"
train_path = OUTPUT_DIR / "train.csv"
val_path = OUTPUT_DIR / "val.csv"
test_path = OUTPUT_DIR / "test.csv"

final_df.to_csv(master_path, index=False)

train_df = final_df[final_df["split"] == "train"]
val_df = final_df[final_df["split"] == "val"]
test_df = final_df[final_df["split"] == "test"]

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Master dataset saved: {master_path}")
print(f"  - Train: {train_path} ({len(train_df)} frames, {len(train_df)/len(final_df):.1%})")
print(f"  - Val:   {val_path}   ({len(val_df)} frames, {len(val_df)/len(final_df):.1%})")
print(f"  - Test:  {test_path}  ({len(test_df)} frames, {len(test_df)/len(final_df):.1%})")


# --------------------------------------------------
# 5. GENERATE CONTROL & DISTRIBUTION PLOTS
# --------------------------------------------------

# Figure 1: Steering, Speed, and Control Distributions across Splits
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Steering Angle Distribution
for split, s_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    axes[0, 0].hist(s_df["steering_angle_deg"], bins=40, alpha=0.5, label=f"{split} (N={len(s_df)})", density=True)
axes[0, 0].set_title("Steering Angle Distribution (deg)")
axes[0, 0].set_xlabel("Steering Angle (°)")
axes[0, 0].set_ylabel("Density")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Speed Distribution
for split, s_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    axes[0, 1].hist(s_df["speed_kmh"], bins=40, alpha=0.5, label=f"{split}", density=True)
axes[0, 1].set_title("Vehicle Speed Distribution (km/h)")
axes[0, 1].set_xlabel("Speed (km/h)")
axes[0, 1].set_ylabel("Density")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Throttle vs Brake Command
axes[1, 0].hist(final_df["throttle_command"], bins=30, alpha=0.7, color="tab:green", label="Throttle", density=True)
axes[1, 0].hist(final_df["brake_command"], bins=30, alpha=0.7, color="tab:red", label="Brake", density=True)
axes[1, 0].set_title("Throttle & Brake Command Distribution")
axes[1, 0].set_xlabel("Command Magnitude [0, 1]")
axes[1, 0].set_ylabel("Density")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Detections vs Speed Scatter
axes[1, 1].scatter(final_df["speed_kmh"], final_df["total_detections"], alpha=0.25, c=final_df["forward_acceleration_mps2"], cmap="coolwarm", s=15)
axes[1, 1].set_title("Speed vs Surrounding Object Detections")
axes[1, 1].set_xlabel("Speed (km/h)")
axes[1, 1].set_ylabel("Total Objects Detected")
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label("Accel (m/s²)")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
dist_plot_path = PLOT_DIR / "control_distributions_and_splits.png"
plt.savefig(dist_plot_path, dpi=200)
plt.close()
print(f"\nSaved distribution plot: {dist_plot_path}")

# Figure 2: Sample Control Profiles over time for sample sequences
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
sample_seq = "2011_09_26_drive_0009_sync"
s_data = final_df[final_df["sequence"] == sample_seq]

t = s_data["elapsed_time_s"]
axes[0].plot(t, s_data["speed_kmh"], color="tab:blue", label="Speed (km/h)")
axes[0].set_ylabel("Speed (km/h)")
axes[0].set_title(f"Sample Control Profile: {sample_seq}")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="upper right")

axes[1].plot(t, s_data["steering_angle_deg"], color="tab:purple", label="Steering Angle (deg)")
axes[1].set_ylabel("Steering (°)")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="upper right")

axes[2].plot(t, s_data["throttle_command"], color="tab:green", label="Throttle [0-1]", alpha=0.8)
axes[2].plot(t, s_data["brake_command"], color="tab:red", label="Brake [0-1]", alpha=0.8)
axes[2].set_ylabel("Command [0, 1]")
axes[2].set_xlabel("Time (seconds)")
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc="upper right")

plt.tight_layout()
profile_plot_path = PLOT_DIR / "sample_sequence_control_profile.png"
plt.savefig(profile_plot_path, dpi=200)
plt.close()
print(f"Saved sample profile plot: {profile_plot_path}")


# --------------------------------------------------
# 6. PRINT SUMMARY TABLE
# --------------------------------------------------

print("\n" + "=" * 55)
print("FINAL AV CONTROL DATASET SUMMARY")
print("=" * 55)
print(f"Total Sequences:        {final_df['sequence'].nunique()}")
print(f"Total Frames:           {len(final_df)}")
print(f"Steering Angle Range:   [{final_df['steering_angle_deg'].min():.2f}°, {final_df['steering_angle_deg'].max():.2f}°]")
print(f"Steering Angle Std:     {final_df['steering_angle_deg'].std():.2f}°")
print(f"Speed Range:            [{final_df['speed_kmh'].min():.1f}, {final_df['speed_kmh'].max():.1f}] km/h (Avg: {final_df['speed_kmh'].mean():.1f} km/h)")
print(f"Throttle > 0 instances: {(final_df['throttle_command'] > 0).sum()} frames")
print(f"Brake > 0 instances:    {(final_df['brake_command'] > 0).sum()} frames")
print(f"Total Objects Detected: {final_df['total_detections'].sum()}")

print("\nSplit Allocation:")
for s_name in ["train", "val", "test"]:
    subset = final_df[final_df["split"] == s_name]
    seq_list = [s.replace("2011_09_26_drive_", "").replace("_sync", "") for s in subset["sequence"].unique()]
    print(f"  {s_name.upper():5s}: {len(subset):4d} frames ({len(subset)/len(final_df):.1%}) | Sequences: {', '.join(seq_list)}")

print("\nDataset preparation complete! Ready for model training.")

