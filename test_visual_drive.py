"""
KITTI Autonomous Driving - Live Visual AI Test Dashboard
Simulates real-time autonomous driving by playing test sequences, overlaying
YOLO perception detections, and visualizing live AI steering and pedal control.

Features:
  - Real-time video playback with bounding boxes for dynamic objects
  - Dual steering wheel visualizer (Human Ground Truth vs AI Model Prediction)
  - Speedometer and live Throttle/Brake control gauges
  - Sub-millisecond latency monitor
  - Interactive playback (Space to pause, 'q' to quit)
  - Auto-export to MP4 video for presentations

Usage:
  python test_visual_drive.py                      # Interactive window
  python test_visual_drive.py --save-video         # Save MP4 video
  python test_visual_drive.py --sequence 0001     # Test drive_0001_sync
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --------------------------------------------------
# 1. SETUP PATHS AND CLASSES
# --------------------------------------------------

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "processed_data"
RAW_DIR = BASE / "raw_data" / "2011_09_26"
RUNS_DIR = BASE / "runs" / "detect"
MODEL_DIR = BASE / "model"
VIS_DIR = BASE / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.pth"
NORM_PATH = MODEL_DIR / "normalisation_params.csv"
TEST_DATA_PATH = DATA_DIR / "test.csv"

COCO_CLASSES = {
    0: ("person", (0, 255, 128)),
    1: ("bicycle", (255, 200, 0)),
    2: ("car", (255, 140, 0)),
    3: ("motorcycle", (255, 255, 0)),
    5: ("bus", (0, 165, 255)),
    7: ("truck", (0, 100, 255)),
    9: ("traffic_light", (255, 0, 255)),
}

INPUT_FEATURES = [
    "speed_kmh", "forward_speed_mps", "heading_deg", "yaw_rate_rad_s",
    "forward_acceleration_mps2", "lateral_acceleration_mps2", "curvature_m_inv",
    "persons_detected", "bicycles_detected", "cars_detected",
    "trucks_detected", "buses_detected", "traffic_lights_detected", "total_detections",
]


# --------------------------------------------------
# 2. MODEL DEFINITION
# --------------------------------------------------

class AVControlModel(nn.Module):
    def __init__(self, input_size=14, hidden_sizes=[64, 32, 16], output_size=3):
        super().__init__()
        layers = []
        prev_size = input_size
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.05))
            prev_size = h_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        self.register_buffer("target_scales", torch.ones(output_size, dtype=torch.float32))

    def forward(self, x):
        return self.network(x) * self.target_scales


# --------------------------------------------------
# 3. DASHBOARD DRAWING HELPERS
# --------------------------------------------------

def draw_steering_gauge(canvas, center_x, center_y, radius, actual_deg, pred_deg):
    """Draws an intuitive instrument cluster dial: Right turns point RIGHT, Left turns point LEFT."""
    # Background dial arc
    cv2.circle(canvas, (center_x, center_y), radius, (45, 45, 45), -1)
    cv2.circle(canvas, (center_x, center_y), radius, (80, 80, 80), 2)
    cv2.circle(canvas, (center_x, center_y), 4, (200, 200, 200), -1)

    # Reference tick marks
    # In automotive displays: Left is Left, Right is Right
    for angle in [-30, -15, 0, 15, 30]:
        rad = np.deg2rad(angle - 90)
        p1 = (int(center_x + (radius - 10) * np.cos(rad)), int(center_y + (radius - 10) * np.sin(rad)))
        p2 = (int(center_x + radius * np.cos(rad)), int(center_y + radius * np.sin(rad)))
        cv2.line(canvas, p1, p2, (120, 120, 120), 1 if angle != 0 else 2)

    # Add 'L' and 'R' direction labels on dial
    cv2.putText(canvas, "L", (center_x - radius + 5, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)
    cv2.putText(canvas, "R", (center_x + radius - 14, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

    # NOTE ON SIGN CONVENTION:
    # KITTI OXTS uses ISO/CCW coordinates: turning Left is (+), turning Right is (-).
    # On an intuitive car dashboard display, turning Right must tilt the needle RIGHT!
    # Therefore, visual angle = -steering_deg.
    vis_actual = -actual_deg
    vis_pred = -pred_deg

    # Human pointer (Green)
    h_rad = np.deg2rad(vis_actual - 90)
    h_tip = (int(center_x + (radius - 14) * np.cos(h_rad)), int(center_y + (radius - 14) * np.sin(h_rad)))
    cv2.line(canvas, (center_x, center_y), h_tip, (50, 220, 50), 3, cv2.LINE_AA)

    # AI pointer (Cyan/Yellow)
    ai_rad = np.deg2rad(vis_pred - 90)
    ai_tip = (int(center_x + (radius - 8) * np.cos(ai_rad)), int(center_y + (radius - 8) * np.sin(ai_rad)))
    cv2.line(canvas, (center_x, center_y), ai_tip, (240, 200, 0), 2, cv2.LINE_AA)

    # Human-readable direction strings: e.g. "10.6 deg (R)" or "15.2 deg (L)"
    def format_steer(deg):
        direction = "R" if deg < -0.3 else ("L" if deg > 0.3 else "C")
        return f"{abs(deg):4.1f} deg ({direction})"

    err_deg = abs(pred_deg - actual_deg)
    base_y = center_y + radius + 18

    cv2.putText(canvas, "STEERING ANGLE", (center_x - 62, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.putText(canvas, f"AI:    {format_steer(pred_deg)}", (center_x - 62, base_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (240, 200, 0), 1)
    cv2.putText(canvas, f"Human: {format_steer(actual_deg)}", (center_x - 62, base_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (50, 220, 50), 1)
    cv2.putText(canvas, f"Error:   {err_deg:4.2f} deg", (center_x - 62, base_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (0, 220, 255), 1)


def draw_vertical_bar(canvas, top_left_x, top_left_y, width, height, value, color, label):
    """Draws a vertical pedal fill bar (0.0 to 1.0)."""
    cv2.rectangle(canvas, (top_left_x, top_left_y), (top_left_x + width, top_left_y + height), (35, 35, 35), -1)
    cv2.rectangle(canvas, (top_left_x, top_left_y), (top_left_x + width, top_left_y + height), (75, 75, 75), 1)

    fill_h = int(np.clip(value, 0.0, 1.0) * height)
    if fill_h > 0:
        cv2.rectangle(canvas, (top_left_x + 1, top_left_y + height - fill_h), (top_left_x + width - 1, top_left_y + height - 1), color, -1)

    cv2.putText(canvas, label, (top_left_x - 5, top_left_y + height + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    cv2.putText(canvas, f"{int(value * 100)}%", (top_left_x - 2, top_left_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)


# --------------------------------------------------
# 4. MAIN DASHBOARD APPLICATION
# --------------------------------------------------

def run_visual_drive(seq_id="0014", save_video=False, no_gui=False, max_frames=None, live_yolo=False):
    # Match sequence name
    seq_name = f"2011_09_26_drive_{seq_id.zfill(4)}_sync"
    seq_raw_dir = RAW_DIR / seq_name
    seq_labels_dir = RUNS_DIR / seq_name / "labels"

    if not seq_raw_dir.exists():
        raise FileNotFoundError(f"Sequence not found: {seq_raw_dir}")

    # Load test dataset row metadata
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Test CSV not found: {TEST_DATA_PATH}")

    df_test = pd.read_csv(TEST_DATA_PATH)
    seq_df = df_test[df_test["sequence"] == seq_name].sort_values("frame_index").reset_index(drop=True)

    if len(seq_df) == 0:
        print(f"Notice: {seq_name} not in test.csv. Falling back to final_av_control_dataset.csv.")
        master_df = pd.read_csv(DATA_DIR / "final_av_control_dataset.csv")
        seq_df = master_df[master_df["sequence"] == seq_name].sort_values("frame_index").reset_index(drop=True)

    total_frames = len(seq_df)
    if max_frames:
        total_frames = min(total_frames, max_frames)
        seq_df = seq_df.iloc[:total_frames]

    print(f"\n=======================================================")
    print(f"STARTING VISUAL DRIVE SIMULATION")
    print(f"Sequence:     {seq_name} ({total_frames} frames)")
    print(f"Perception:   {'Live YOLO Inference' if live_yolo else 'Cached YOLO Detections'}")
    print(f"Control AI:   100% Live Neural Network Inference")
    print(f"Mode:         {'Video Export' if save_video else 'Live Interactive'}")
    print(f"=======================================================\n")

    # Load Model and Normalisation
    norm_df = pd.read_csv(NORM_PATH).set_index("feature")
    mean_vals = norm_df.loc[INPUT_FEATURES, "mean"].values.astype(np.float32)
    std_vals = norm_df.loc[INPUT_FEATURES, "std"].values.astype(np.float32)

    model = AVControlModel(len(INPUT_FEATURES), [64, 32, 16], 3)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location="cpu"))
    model.eval()

    yolo_model = None
    if live_yolo:
        from ultralytics import YOLO
        yolo_model = YOLO("yolo26n.pt")
        print("✓ Live YOLO model loaded.")

    # Target dashboard dimensions
    VIEW_W, VIEW_H = 1242, 375
    HUD_H = 210
    CANVAS_W, CANVAS_H = VIEW_W, VIEW_H + HUD_H

    # Video writer setup if saving
    video_writer = None
    output_video_path = VIS_DIR / f"ai_drive_{seq_id}.mp4"
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, 10.0, (CANVAS_W, CANVAS_H))
        print(f"Recording video to: {output_video_path}")

    # Interactive window setup
    window_name = f"KITTI Autonomous Vehicle Edge AI Controller - {seq_name}"
    if not no_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1242, CANVAS_H)

    paused = False
    delay_ms = 40  # nominal ~25 fps playback

    for i in range(total_frames):
        row = seq_df.iloc[i]
        frame_idx = int(row["frame_index"])
        img_path = seq_raw_dir / "image_02" / "data" / f"{frame_idx:010d}.png"

        if not img_path.exists():
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        h_orig, w_orig = frame.shape[:2]

        # Object detections (Live YOLO or Pre-saved labels)
        det_counts = {"car": 0, "person": 0, "truck": 0, "traffic_light": 0}

        if live_yolo and yolo_model is not None:
            results = yolo_model(frame, verbose=False, imgsz=640)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                cls_name, color = COCO_CLASSES.get(cls_id, ("object", (200, 200, 200)))
                if cls_name in det_counts:
                    det_counts[cls_name] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                tag = f"{cls_name} {int(conf * 100)}%"
                cv2.putText(frame, tag, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        else:
            # Draw YOLO detection boxes from saved labels
            label_file = seq_labels_dir / f"{frame_idx:010d}.txt"
            if label_file.exists():
                with open(label_file, "r") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(float(parts[0]))
                            cx, cy, bw, bh = [float(v) for v in parts[1:5]]
                            conf = float(parts[5]) if len(parts) > 5 else 1.0

                            # Convert normalized bbox to pixels
                            x1 = int((cx - bw / 2) * w_orig)
                            y1 = int((cy - bh / 2) * h_orig)
                            x2 = int((cx + bw / 2) * w_orig)
                            y2 = int((cy + bh / 2) * h_orig)

                            cls_name, color = COCO_CLASSES.get(cls_id, ("object", (200, 200, 200)))
                            if cls_name in det_counts:
                                det_counts[cls_name] += 1

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            tag = f"{cls_name} {int(conf * 100)}%"
                            cv2.putText(frame, tag, (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        # Run AI model inference
        sensor_input = row[INPUT_FEATURES].values.astype(np.float32)
        norm_input = (sensor_input - mean_vals) / std_vals
        tensor_in = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0)

        t_start = time.perf_counter()
        with torch.no_grad():
            preds = model(tensor_in).squeeze(0).numpy()
        inf_latency_ms = (time.perf_counter() - t_start) * 1000

        pred_steering = float(preds[0])
        pred_throttle = float(np.clip(preds[1], 0.0, 1.0))
        pred_brake = float(np.clip(preds[2], 0.0, 1.0))

        actual_steering = float(row["steering_angle_deg"])
        actual_throttle = float(row["throttle_command"])
        actual_brake = float(row["brake_command"])
        speed_kmh = float(row["speed_kmh"])

        # Compose full canvas (Camera view on top, HUD telemetry at bottom)
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[:VIEW_H, :VIEW_W] = frame

        # Top overlay banner
        cv2.rectangle(canvas, (0, 0), (CANVAS_W, 35), (20, 20, 20), -1)
        cv2.putText(canvas, f"KITTI: {seq_name} | Frame: {frame_idx:04d}/{total_frames:04d}", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(canvas, "DISTRIBUTED EDGE AI CONTROL: ACTIVE", (CANVAS_W - 390, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2)

        # Bottom HUD background
        hud_y = VIEW_H
        cv2.rectangle(canvas, (0, hud_y), (CANVAS_W, CANVAS_H), (15, 15, 15), -1)
        cv2.line(canvas, (0, hud_y), (CANVAS_W, hud_y), (60, 60, 60), 2)

        # 1. Speedometer Section (Left)
        cv2.putText(canvas, f"{speed_kmh:.1f}", (45, hud_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3)
        cv2.putText(canvas, "km/h", (48, hud_y + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 160, 160), 1)
        cv2.putText(canvas, f"Accel: {row['forward_acceleration_mps2']:+.2f} m/s^2", (45, hud_y + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(canvas, f"Yaw Rate: {np.rad2deg(row['yaw_rate_rad_s']):+.2f} deg/s", (45, hud_y + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.line(canvas, (240, hud_y + 20), (240, CANVAS_H - 20), (45, 45, 45), 1)

        # 2. Steering Instrument Cluster (Center-Left)
        draw_steering_gauge(canvas, center_x=380, center_y=hud_y + 70, radius=48, actual_deg=actual_steering, pred_deg=pred_steering)

        cv2.line(canvas, (520, hud_y + 20), (520, CANVAS_H - 20), (45, 45, 45), 1)

        # 3. Throttle & Brake Bars (Center-Right)
        bar_y = hud_y + 40
        bar_h = 100
        # Throttle bars
        draw_vertical_bar(canvas, top_left_x=570, top_left_y=bar_y, width=22, height=bar_h, value=actual_throttle, color=(0, 180, 0), label="THR(H)")
        draw_vertical_bar(canvas, top_left_x=610, top_left_y=bar_y, width=22, height=bar_h, value=pred_throttle, color=(0, 255, 128), label="THR(AI)")

        # Brake bars
        draw_vertical_bar(canvas, top_left_x=680, top_left_y=bar_y, width=22, height=bar_h, value=actual_brake, color=(0, 0, 180), label="BRK(H)")
        draw_vertical_bar(canvas, top_left_x=720, top_left_y=bar_y, width=22, height=bar_h, value=pred_brake, color=(0, 80, 255), label="BRK(AI)")

        cv2.line(canvas, (785, hud_y + 20), (785, CANVAS_H - 20), (45, 45, 45), 1)

        # 4. Perception & Edge AI Performance (Right)
        cv2.putText(canvas, "PERCEPTION & EDGE METRICS", (820, hud_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
        cv2.putText(canvas, f"Cars in scene:       {det_counts['car']}", (820, hud_y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 140, 0), 1)
        cv2.putText(canvas, f"Pedestrians:         {det_counts['person']}", (820, hud_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 128), 1)
        cv2.putText(canvas, f"Trucks / Buses:      {det_counts['truck'] + det_counts.get('bus', 0)}", (820, hud_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1)

        cv2.putText(canvas, f"Model Inference:     {inf_latency_ms:.2f} ms", (820, hud_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 220), 1)
        cv2.putText(canvas, f"Edge Loop Rate:      ~{int(1000.0 / max(inf_latency_ms, 0.05)):,} Hz", (820, hud_y + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 220), 1)

        # Write frame to video
        if video_writer is not None:
            video_writer.write(canvas)

        # Display interactive window
        if not no_gui:
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("Simulation stopped by user.")
                break
            elif key == ord(' '):  # Spacebar pause/resume
                paused = not paused
            elif key == ord('+') or key == ord('='):
                delay_ms = max(5, delay_ms - 10)
            elif key == ord('-'):
                delay_ms = min(200, delay_ms + 10)

    if video_writer is not None:
        video_writer.release()
        print(f"\n✓ Video simulation successfully saved to: {output_video_path}")

    if not no_gui:
        cv2.destroyAllWindows()

    print("\nVisual simulation completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KITTI Autonomous Vehicle Visual Test Dashboard")
    parser.add_argument("--sequence", type=str, default="0014", help="Sequence ID (e.g. 0014, 0001, 0020)")
    parser.add_argument("--save-video", action="store_true", help="Record and save full MP4 simulation video")
    parser.add_argument("--no-gui", action="store_true", help="Run without opening window (for background/testing)")
    parser.add_argument("--live-yolo", action="store_true", help="Run live YOLO inference on each frame instead of using saved labels")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to run")

    args = parser.parse_args()
    run_visual_drive(
        seq_id=args.sequence,
        save_video=args.save_video,
        no_gui=args.no_gui,
        max_frames=args.max_frames,
        live_yolo=args.live_yolo
    )

