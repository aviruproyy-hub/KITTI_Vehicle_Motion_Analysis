# Distributed Edge AI System for Autonomous Vehicle Motion Control & Real-Time Simulation
## Comprehensive Technical Architecture, Data Engineering & Verification Report

**Author:** Avirup Roy & Autonomous Systems Team  
**Date:** November 2026  
**Repository:** `KITTI_Vehicle_Motion_Analysis`  
**Target Hardware:** Automotive Embedded ECUs (NVIDIA Jetson, NXP S32G, Renesas R-Car)  

---

## 1. Executive Summary & Abstract

Autonomous ground vehicles operating in dynamic real-world environments require high-frequency, deterministic, and fail-safe motion control systems. While deep reinforcement learning and complex end-to-end vision-language models offer impressive capabilities in research simulations, their substantial computational footprint (often exceeding 50 ms per forward pass) makes them unsuitable for strict automotive Electronic Control Units (ECUs) operating under strict power and thermal budgets ($10 - 30\,\text{W}$).

This project presents an end-to-end, safety-critical **Distributed Edge AI System for Autonomous Vehicle Motion Control**. The pipeline ingests synchronized sensor telemetry from real-world **KITTI raw driving sequences** (inertial measurement units, high-precision GPS, longitudinal velocities, and 3D perception bboxes), extracts physical motion primitives, and trains an ultra-compact multi-task neural network controller (`AVControlModel`).

```
+---------------------------------------------------------------------------------------------------+
|                                  SYSTEM ARCHITECTURE OVERVIEW                                      |
|                                                                                                   |
|  [KITTI OXTS / RT3003 GPS/IMU] + [YOLO 3D Perception]                                             |
|                             │                                                                     |
|                             ▼                                                                     |
|  Feature Engineering: Curvature (κ), Lateral Accel (ay), Longitudinal Speeds (v), Lookahead (ld)  |
|                             │                                                                     |
|                             ▼                                                                     |
|  Z-Score Normalization Pipeline (mean μ, standard deviation σ)                                    |
|                             │                                                                     |
|                             ▼                                                                     |
|  Edge Neural Controller (AVControlModel: 14 inputs ➔ 64 ➔ 32 ➔ 16 ➔ 3 outputs)                   |
|                             │                                                                     |
|              ┌──────────────┴──────────────┐                                                      |
|              ▼                             ▼                                                      |
|  2D Closed-Loop Highway Simulator   3D Real-Time Physics Simulator (Ursina/Panda3D)               |
|  - ACC & Multi-Lane Overtaking      - Ackermann Kinematic Geometry (L = 2.7m)                     |
|  - Zero-Collision up to 200 km/h    - Sweeping Bends, S-Chicanes & 90° Intersection               |
|  - Gap Acceptance Safety Filters    - Multi-Cam HUD & Sub-Millisecond Benchmark (0.18 ms)          |
+---------------------------------------------------------------------------------------------------+
```

### Key Quantitative Achievements
- **Sub-Millisecond Inference Latency:** Mean forward-pass latency of **$0.18\,\text{ms}$** (throughput $> 5,500\,\text{Hz}$ on standard CPU), satisfying automotive safety cycles ($20 - 50\,\text{Hz}$) with an order-of-magnitude margin.
- **Precision Path Tracking:** Maximum cross-track lateral error of **$0.464\,\text{m}$** across a complex $623.8\,\text{m}$ test track with sweeping bends ($R=75\,\text{m}$), S-chicanes ($R=55\,\text{m}$), and a sharp $90^\circ$ intersection turn ($R=25\,\text{m}$) on a $7.5\,\text{m}$ road.
- **Closed-Loop High-Speed Stability:** Zero collisions recorded across multi-lane highway overtaking trials at speeds up to $200\,\text{km/h}$.
- **Lightweight Deployability:** Standalone neural weights of only **$18.5\,\text{KB}$** (`.pth`), with native export to **TorchScript** (`.pt`) and **ONNX** (`.onnx`) for direct deployment onto automotive RTOS and TensorRT.

---

## 2. 3D Simulation Showcase

The autonomous vehicle was validated inside a custom-engineered 3D simulation engine running at **60 FPS** on CPU and integrated Intel graphics without requiring heavy external simulators such as CARLA.

![3D Simulation Showcase](visualizations/3d_sim_showcase.png)

*Figure 1: Closed-loop autonomous vehicle navigation in the 3D environment across 4 perspectives: (Top-Left) 3rd-Person Chase Cam, (Top-Right) 1st-Person Cockpit/Hood Cam, (Bottom-Left) Overhead Drone Cam (38m altitude), (Bottom-Right) Arrival at Point B Finish Gantry.*

---

## 3. Repository & Source Code Architecture

The repository is structured following automotive software engineering standards:

```text
KITTI_Vehicle_Motion_Analysis/
├── model/
│   ├── best_model.pth              # PyTorch state_dict neural network weights (Primary)
│   ├── normalisation_params.csv     # Z-score normalization mean & std per feature
│   ├── target_scales.csv            # Target scaling standard deviations
│   ├── av_controller.pt             # Compiled TorchScript JIT model for C++ runtimes
│   └── av_controller.onnx           # ONNX model for TensorRT/OpenVINO edge hardware
├── processed_data/
│   ├── final_av_control_dataset.csv # Full consolidated tabular dataset (35 attributes)
│   ├── train.csv                    # Training partition (70% split)
│   ├── val.csv                      # Validation partition (15% split)
│   ├── test.csv                     # Held-out testing partition (15% split)
│   ├── kitti_vehicle_motion.csv     # Extracted physical kinematics from OXTS
│   └── combined_motion_detections.csv# Feature-fused dataset with perception objects
├── visualizations/
│   ├── 3d_sim_showcase.png          # 4-angle visual composite of 3D simulation
│   ├── dashboard_preview.png        # Live perception replayer dashboard preview
│   ├── training_results.png         # Loss and validation curves across 40 epochs
│   ├── control_distributions_and_splits.png # Target distributions and train/val/test splits
│   ├── sample_sequence_control_profile.png  # Ground truth vs predicted control profiles
│   ├── speed_over_time.png          # Temporal velocity profiles across sequences
│   ├── heading_over_time.png        # Heading angle tracking
│   ├── trajectory_xy.png            # Planar trajectory traces
│   ├── ai_drive_0014.mp4            # Real-time recorded video of autonomous urban run
│   └── autonomous_highway_drive.mp4 # Video of high-speed multi-lane overtaking
├── simulate_3d_drive.py             # Interactive 3D Autonomous Vehicle Simulator (Ursina/Panda3D)
├── simulate_highway_drive.py        # 2D Multi-Lane Highway Overtaking Simulator
├── predict_control.py               # Edge AI Real-Time Inference Engine & Benchmark Class
├── test_sim_physics.py              # Headless physics & pure pursuit tracking unit test
├── train_model.py                   # PyTorch multi-task training pipeline
├── prepare_control_dataset.py       # Normalization calculation and train/val/test split generator
├── process_kitti.py                 # Raw KITTI OXTS extraction and trajectory kinematics parser
├── combine_detections.py            # Feature fusion: merges YOLO perception with motion data
├── run_yolo.py                      # YOLO perception detection utility
├── export_onnx.py                   # Model exporter for ONNX and TorchScript formats
├── requirements.txt                 # Exact Python dependency specification
├── README.md                        # Documentation and GitHub project guide
└── .gitignore                       # Git exclusions (raw data >13GB, caches, temporary runs)
```

### Detailed Script Functionality Guide

1. **`simulate_3d_drive.py`**:
   The interactive 3D simulation application built with Panda3D and Ursina. It instantiates the analytical $C^1$-smooth highway and urban track, renders 3D cityscape architecture (buildings, trees, lamp posts, gantries), models vehicle body chassis and swiveling Ackermann steering wheels, simulates rolling tire physics and dynamic red brake lights, renders an interactive telemetry HUD, and hosts a 3-way toggleable camera system (Chase Cam, Cockpit/Hood Cam, Overhead Drone Cam).
2. **`simulate_highway_drive.py`**:
   A closed-loop 2D multi-lane highway environment simulating high-speed driving ($100 - 200\,\text{km/h}$). It features dynamic obstacle vehicles, radar lookahead distance monitoring, adaptive cruise control (ACC), lane selection, and overtaking maneuvers.
3. **`predict_control.py`**:
   The production-grade Edge AI inference class (`EdgeVehicleController`). It encapsulates feature normalization, neural forward pass, target de-scaling, and fallback heuristics with an inference latency of $0.18\,\text{ms}$.
4. **`test_sim_physics.py`**:
   A headless pure-pursuit and vehicle dynamics verification script. It runs 6,000 continuous simulation steps in memory in under 2 seconds, printing distance progress, speed tracking, lateral error bounds, and final arrival status.
5. **`train_model.py`**:
   The core PyTorch model training script. It loads the normalized dataset, builds `AVControlModel`, executes 40 training epochs with the Adam optimizer and target-scaled loss, records validation metrics, and saves `model/best_model.pth`.
6. **`prepare_control_dataset.py`**:
   Reads `processed_data/final_av_control_dataset.csv`, calculates mean ($\mu$) and standard deviation ($\sigma$) parameters for input z-score normalization, determines target scaling factors, and partitions the dataset into stratified `train.csv`, `val.csv`, and `test.csv`.
7. **`process_kitti.py`**:
   The sensor extraction engine. Parses raw binary/ASCII OXTS packets from the KITTI dataset, transforms geodetic GPS coordinates into local Cartesian East-North-Up (ENU) coordinates via Mercator projection, calculates forward velocity, acceleration, and yaw rate, and derives road curvature.

---

## 4. Data Engineering & Complete CSV Attribute Dictionary

The dataset was constructed from the benchmark **KITTI Vision Benchmark Suite** collected by the Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago. The vehicle platform was an instrumented Volkswagen Passat station wagon equipped with:
- An **OXTS RT3003** navigation system fusing a 6-DOF Inertial Measurement Unit (IMU) and a high-accuracy GPS receiver at $100\,\text{Hz}$.
- Two PointGrey Flea2 color stereo cameras ($1392 \times 512$ resolution at $10\,\text{Hz}$).
- A Velodyne HDL-64E rotating 3D LiDAR scanner ($10\,\text{Hz}$).

```
+───────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    KITTI SENSOR HARVESTING                                        |
|                                                                                                   |
|  [OXTS RT3003 GPS/IMU (100Hz)] ───────► Pos (lat, lon, alt) + Vel (vx, vy) + Accel (ax, ay)      |
|                                          │                                                        |
|  [PointGrey Stereo Cam (10Hz)] ───────► YOLO Detections (cars, pedestrians, trucks, lights)       |
|                                          │                                                        |
|                                          ▼                                                        |
|                             [process_kitti.py Parser]                                             |
|                                          │                                                        |
|                 ┌────────────────────────┴────────────────────────┐                               |
|                 ▼                                                 ▼                               |
|       Mercator Projection (x, y)                       Kinematic Derivations                      |
|       - Easting (x_m)                                  - Curvature: κ = dψ/ds                     |
|       - Northing (y_m)                                 - Steer: δ = arctan(L·κ)                   |
|       - Altitude (alt_m)                               - Thr / Brk Mapping                        |
|                 │                                                 │                               |
|                 └────────────────────────┬────────────────────────┘                               |
|                                          ▼                                                        |
|                       processed_data/final_av_control_dataset.csv                                 |
+───────────────────────────────────────────────────────────────────────────────────────────────────+
```

### Complete Attribute Dictionary (All 35 Columns)

Every row in `processed_data/final_av_control_dataset.csv` represents a synchronized discrete time sample ($t$) with the following 35 attributes:

| # | Attribute Column Name | Data Type | Physical Units | Sensor Source | Model Role | Mathematical & Physical Description |
| :-: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | `sequence` | `string` | — | Metadata | Identifier | Raw KITTI sequence directory name (e.g. `2011_09_26_drive_0001_sync`). |
| **2** | `frame_index` | `int64` | index | Metadata | Identifier | Zero-indexed temporal frame counter within the sequence. |
| **3** | `timestamp` | `string` | UTC ISO8601 | OXTS GPS | Metadata | High-precision microsecond UTC timestamp of the sensor measurement. |
| **4** | `elapsed_time_s` | `float64` | $\text{seconds}$ | Derived | Metadata | Cumulative elapsed time since sequence start: $\Delta t = t_i - t_0$. |
| **5** | `latitude` | `float64` | $\text{degrees}$ | OXTS GPS | Coordinate | WGS-84 geodetic latitude coordinate. |
| **6** | `longitude` | `float64` | $\text{degrees}$ | OXTS GPS | Coordinate | WGS-84 geodetic longitude coordinate. |
| **7** | `altitude_m` | `float64` | $\text{meters}$ | OXTS GPS | Coordinate | Height above mean sea level (WGS-84 ellipsoid). |
| **8** | `x_m` | `float64` | $\text{meters}$ | Derived | Coordinate | Local Cartesian Easting coordinate relative to sequence starting point $(0,0)$. |
| **9** | `y_m` | `float64` | $\text{meters}$ | Derived | Coordinate | Local Cartesian Northing coordinate relative to sequence starting point $(0,0)$. |
| **10** | `roll_rad` | `float64` | $\text{radians}$ | OXTS IMU | State | Vehicle rotation angle around the longitudinal body axis (bank angle). |
| **11** | `pitch_rad` | `float64` | $\text{radians}$ | OXTS IMU | State | Vehicle rotation angle around the lateral body axis (elevation angle). |
| **12** | `heading_yaw_rad` | `float64` | $\text{radians}$ | OXTS IMU | State | Vehicle heading angle relative to geographic East in radians: $\psi \in [-\pi, \pi]$. |
| **13** | `heading_deg` | `float64` | $\text{degrees}$ | Derived | **Input Feature** | Vehicle heading converted to degrees: $\psi_{\text{deg}} = \psi \times \frac{180}{\pi}$. |
| **14** | `speed_mps` | `float64` | $\text{m/s}$ | OXTS IMU | State | Ground velocity magnitude: $v = \sqrt{v_x^2 + v_y^2}$. |
| **15** | `speed_kmh` | `float64` | $\text{km/h}$ | Derived | **Input Feature** | Instantaneous vehicle speed in $\text{km/h}$: $v_{\text{km/h}} = v \times 3.6$. |
| **16** | `forward_speed_mps` | `float64` | $\text{m/s}$ | OXTS IMU | **Input Feature** | Longitudinal velocity along the vehicle's forward heading axis ($v_x$). |
| **17** | `forward_acceleration_mps2` | `float64` | $\text{m/s}^2$ | OXTS IMU | **Input Feature** | Longitudinal acceleration along forward axis ($a_x$). Positive = accel, negative = braking. |
| **18** | `lateral_acceleration_mps2` | `float64` | $\text{m/s}^2$ | OXTS IMU | **Input Feature** | Centripetal acceleration along lateral axis ($a_y = v \cdot \dot{\psi}$). |
| **19** | `persons_detected` | `int64` | count | YOLO Vision | **Input Feature** | Count of detected pedestrian bounding boxes in the front camera field of view. |
| **20** | `bicycles_detected` | `int64` | count | YOLO Vision | **Input Feature** | Count of detected cyclist bounding boxes in the front camera field of view. |
| **21** | `cars_detected` | `int64` | count | YOLO Vision | **Input Feature** | Count of detected passenger cars in the front camera field of view. |
| **22** | `motorcycles_detected` | `int64` | count | YOLO Vision | Context | Count of detected motorcycles. |
| **23** | `buses_detected` | `int64` | count | YOLO Vision | **Input Feature** | Count of detected transit buses. |
| **24** | `trucks_detected` | `int64` | count | YOLO Vision | **Input Feature** | Count of detected heavy trucks and commercial transport vehicles. |
| **25** | `traffic_lights_detected` | `int64` | count | YOLO Vision | **Input Feature** | Count of detected traffic light fixtures. |
| **26** | `total_detections` | `int64` | count | YOLO Vision | **Input Feature** | Aggregate sum of all dynamic traffic participants in the immediate environment. |
| **27** | `other_detections` | `int64` | count | YOLO Vision | Context | Miscellaneous unclassified obstacle detections. |
| **28** | `yaw_rate_rad_s` | `float64` | $\text{rad/s}$ | OXTS IMU | **Input Feature** | Rotational angular velocity around vertical axis: $\dot{\psi} = \frac{d\psi}{dt}$. |
| **29** | `steering_angle_rad` | `float64` | $\text{radians}$ | Derived | Kinematic | Ackermann front wheel angle: $\delta = \arctan\left(\frac{L \dot{\psi}}{v}\right)$. |
| **30** | `steering_angle_deg` | `float64` | $\text{degrees}$ | Derived | **Target Label 1** | Ground truth front wheel steering angle in degrees ($\pm 30^\circ$). |
| **31** | `curvature_m_inv` | `float64` | $\text{m}^{-1}$ | Derived | **Input Feature** | Trajectory spatial curvature: $\kappa = \frac{\dot{\psi}}{v} = \frac{1}{R}$. |
| **32** | `target_acceleration_mps2` | `float64` | $\text{m/s}^2$ | Derived | Target Ref | Filtered longitudinal target acceleration profile. |
| **33** | `throttle_command` | `float64` | $[0.0, 1.0]$ | Derived | **Target Label 2** | Normalized accelerator pedal request: $\text{thr} = \text{clip}\left(\frac{a_x}{a_{\text{max}}}, 0, 1\right)$. |
| **34** | `brake_command` | `float64` | $[0.0, 1.0]$ | Derived | **Target Label 3** | Normalized service brake pedal request: $\text{brk} = \text{clip}\left(\frac{-a_x}{a_{\text{brk,max}}}, 0, 1\right)$. |
| **35** | `split` | `string` | — | Dataset Partition | Split Tag | Partition designation: `train` (70%), `val` (15%), or `test` (15%). |

---

## 5. Mathematical Formulations & Vehicle Kinematics

### 5.1 Kinematic Bicycle Model & Ackermann Geometry

At low-to-moderate lateral acceleration ($a_y < 0.4g$), tire slip angles are negligible. The vehicle dynamics are governed by the planar **Kinematic Bicycle Model**:

$$\dot{x} = v \cos\psi$$
$$\dot{y} = v \sin\psi$$
$$\dot{\psi} = \frac{v}{L} \tan\delta$$

Where:
- $(x, y)$ represents the planar Cartesian position of the center of the rear axle.
- $\psi$ is the inertial orientation (heading) angle of the vehicle.
- $v$ is the longitudinal forward velocity ($v = \dot{s}$).
- $L$ is the physical wheelbase of the vehicle ($L = 2.7\,\text{m}$ for the test Volkswagen Passat).
- $\delta$ is the effective steering angle of the front wheels.

Rearranging the kinematic equation yields the exact relationship between the instantaneous yaw rate $\dot{\psi}$, vehicle velocity $v$, and front steering angle $\delta$:

$$\tan\delta = \frac{L \dot{\psi}}{v} \implies \delta = \arctan\left(\frac{L \dot{\psi}}{\max(v, v_{\min})}\right)$$

```
                           FRONT AXLE (Steered)
                                 ▲
                                /  δ (Steering Angle)
                               /
                              ┌───┐
                              │   │
                              └───┘
                                │
                                │
                                │  Wheelbase L = 2.7 m
                                │
                                │
                              ┌───┐
                              │   │
                              └───┘
                                ▲
                            REAR AXLE (Fixed Heading ψ)
```

### 5.2 Trajectory Curvature Formulation

The curvature $\kappa$ of the road path at any point is defined as the rate of change of tangential heading with respect to arc length $s$:

$$\kappa = \frac{d\psi}{ds} = \frac{d\psi / dt}{ds / dt} = \frac{\dot{\psi}}{v} = \frac{1}{R}$$

Where $R$ is the instantaneous radius of curvature of the turn. In accordance with the KITTI and SAE standard coordinate conventions:
- **Left turn:** $\dot{\psi} > 0 \implies \kappa > 0$
- **Right turn:** $\dot{\psi} < 0 \implies \kappa < 0$
- **Straight road:** $\dot{\psi} = 0 \implies \kappa = 0, R \to \infty$

### 5.3 Pure Pursuit Geometric Path Tracking

To track an arbitrary reference trajectory $C(s)$, the autonomous controller determines an adaptive lookahead distance $l_d$ as a dynamic function of forward velocity:

$$l_d = \text{clip}\left(l_{d0} + k_v \cdot v, \, l_{d,\min}, \, l_{d,\max}\right)$$

With calibrated parameters:
$$l_{d0} = 5.0\,\text{m}, \quad k_v = 0.45\,\text{s}, \quad l_{d,\min} = 6.0\,\text{m}, \quad l_{d,\max} = 16.0\,\text{m}$$

Given the vehicle position $(x, z)$ and the target lookahead waypoint $(x_{\text{look}}, z_{\text{look}})$, the relative displacement vector in the vehicle's local body coordinate frame is:

$$\Delta x_b = (x_{\text{look}} - x) \cos\psi - (z_{\text{look}} - z) \sin\psi$$
$$\Delta z_b = (x_{\text{look}} - x) \sin\psi + (z_{\text{look}} - z) \cos\psi$$

The angle $\alpha$ between the vehicle heading vector and the lookahead vector is:

$$\alpha = \arctan2(\Delta x_b, \Delta z_b)$$

Applying the geometric sine rule on the resulting circle arc connecting the rear axle to the lookahead waypoint yields the closed-form Pure Pursuit steering angle command:

$$\delta_{\text{target}} = \arctan\left(\frac{2 L \sin\alpha}{l_d}\right)$$

The signed cross-track lateral error $e_{\text{lat}}$ relative to the nearest path waypoint $(x_{\text{wp}}, z_{\text{wp}})$ with path tangent angle $\psi_{\text{wp}}$ is:

$$e_{\text{lat}} = (x - x_{\text{wp}}) \cos\psi_{\text{wp}} - (z - z_{\text{wp}}) \sin\psi_{\text{wp}}$$

### 5.4 Curvature-Governed Cornering Velocity

To prevent vehicle rollover and loss of tire traction, the lateral acceleration experienced during cornering must not exceed the passenger comfort threshold $a_{\text{lat,max}} \approx 2.5\,\text{m/s}^2$:

$$a_{\text{lat}} = \frac{v^2}{R} = v^2 |\kappa| \le a_{\text{lat,max}}$$

Solving for maximum safe cornering velocity $v_{\text{curve}}$:

$$v_{\text{curve}} = \sqrt{\frac{a_{\text{lat,max}}}{|\kappa|}} = \sqrt{\frac{2.5}{\max(|\kappa_{\text{look}}|, 10^{-4})}}$$

The overall longitudinal speed target is then clamped by the cruising speed and destination braking distance:

$$v_{\text{target}} = \min\left(v_{\text{cruise}}, \, v_{\text{curve}}, \, v_{\text{stop}}(s_{\text{remain}})\right)$$

### 5.5 Longitudinal Vehicle Dynamics & Resistance Forces

The net longitudinal acceleration $a_{\text{net}}$ reflects engine propulsion, service braking, aerodynamic drag, and tire rolling resistance:

$$F_{\text{net}} = F_{\text{engine}} - F_{\text{brake}} - F_{\text{drag}} - F_{\text{rolling}}$$
$$a_{\text{net}} = a_{\text{engine}}(\text{thr}) + a_{\text{brake}}(\text{brk}) + a_{\text{drag}}(v) + a_{\text{rolling}}$$

Where:
$$a_{\text{engine}} = 3.8 \cdot \text{throttle} \quad (\text{m/s}^2)$$
$$a_{\text{brake}} = -6.5 \cdot \text{brake} \quad (\text{m/s}^2)$$
$$a_{\text{resistance}} = -\left(c_{rr} g + c_d v^2\right) = -\left(0.015 \times 9.81 + 0.0004 v^2\right)$$

Integration of vehicle state over discrete simulation timestep $\Delta t = 0.02\,\text{s}$ ($50\,\text{Hz}$):

$$v_{t+1} = \max\left(0, \, v_t + a_{\text{net}} \Delta t\right)$$
$$\psi_{t+1} = \psi_t + \left(\frac{v_t}{L} \tan\delta\right) \Delta t$$
$$x_{t+1} = x_t + v_t \sin\psi_t \Delta t$$
$$z_{t+1} = z_t + v_t \cos\psi_t \Delta t$$

---

## 6. Neural Network Architecture & Balanced Multi-Target Loss

```
INPUT LAYER (14 Features)
┌────────────────────────────────────────────────────────┐
│ speed_kmh, forward_speed_mps, heading_deg,             │
│ yaw_rate_rad_s, forward_accel_mps2, lateral_accel_mps2,│
│ curvature_m_inv, persons_detected, bicycles_detected,  │
│ cars_detected, trucks_detected, buses_detected,        │
│ traffic_lights_detected, total_detections              │
└──────────────────────────┬─────────────────────────────┘
                           │  Z-Score Normalization: z = (x - μ) / σ
                           ▼
                  ┌──────────────────┐
                  │ Linear(14 -> 64) │
                  │ ReLU + Drop(0.05)│
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │ Linear(64 -> 32) │
                  │ ReLU + Drop(0.05)│
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │ Linear(32 -> 16) │
                  │ ReLU + Drop(0.05)│
                  └────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │ Linear(16 -> 3)  │
                  └────────┬─────────┘
                           │  Target De-scaling: y_hat = raw * S
                           ▼
OUTPUT LAYER (3 Vehicle Commands)
┌────────────────────────────────────────────────────────┐
│ 1. steering_angle_deg (±30.0°)                         │
│ 2. throttle_command   (0.0 - 1.0)                      │
│ 3. brake_command      (0.0 - 1.0)                      │
└────────────────────────────────────────────────────────┘
```

### 6.1 Target Scaling Formulation

A fundamental challenge in multi-task regression for autonomous vehicles is **target gradient dominance**. The steering angle $\delta$ varies between $-30^\circ$ and $+30^\circ$ (variance $\sigma_\delta^2 \approx 29.2$), whereas throttle and brake values reside in $[0, 1]$ (variances $\sigma_{\text{thr}}^2 \approx 0.014, \sigma_{\text{brk}}^2 \approx 0.0076$). 

Without target normalization, the unscaled Mean Squared Error loss:

$$\mathcal{L}_{\text{unscaled}} = (\hat{\delta} - \delta)^2 + (\hat{\text{thr}} - \text{thr})^2 + (\hat{\text{brk}} - \text{brk})^2$$

results in steering gradients dominating backpropagation by a ratio of $> 2000 : 1$, completely destroying the network's ability to learn throttle and brake control.

To resolve this, we registered a persistent scaling buffer $S = [S_\delta, S_{\text{thr}}, S_{\text{brk}}]^T$ derived from target standard deviations in `model/target_scales.csv`:

$$S_\delta = 5.4070, \quad S_{\text{thr}} = 0.1198, \quad S_{\text{brk}} = 0.0873$$

The normalized network prediction $\tilde{y} = f_\theta(z)$ is converted to physical units inside the forward pass:

$$\hat{y} = \tilde{y} \odot S$$

The balanced loss function is:

$$\mathcal{L}(\theta) = \frac{1}{B} \sum_{k=1}^B \sum_{j=1}^3 w_j \left( \frac{\hat{y}_{k,j} - y_{k,j}}{S_j} \right)^2$$

Where $w = [1.0, 1.0, 1.0]$. This guarantees equal gradient contributions from all three actuators.

### 6.2 Training Dynamics & Evaluation

![Training Results](visualizations/training_results.png)

*Figure 2: Training and validation loss curves across 40 epochs showing stable convergence without overfitting, alongside validation MAE across all three control outputs.*

---

## 7. Edge AI Benchmarks & Hardware Deployment

### 7.1 Measured Inference Latency Profile

Inference benchmarks were conducted on an Intel Core i5 edge-representative CPU:

| Execution Engine | Mean Latency | Median Latency | 99th Percentile Latency | Control Frequency |
| :--- | :---: | :---: | :---: | :---: |
| **PyTorch (`best_model.pth`)** | **$0.18\,\text{ms}$** | **$0.17\,\text{ms}$** | **$0.31\,\text{ms}$** | **$5,550\,\text{Hz}$** |
| **TorchScript (`av_controller.pt`)** | **$0.14\,\text{ms}$** | **$0.13\,\text{ms}$** | **$0.24\,\text{ms}$** | **$7,140\,\text{Hz}$** |
| **ONNX Runtime (`av_controller.onnx`)**| **$0.09\,\text{ms}$** | **$0.08\,\text{ms}$** | **$0.16\,\text{ms}$** | **$11,100\,\text{Hz}$** |

### 7.2 Automotive Safety Timing Margin

Typical automotive control loops (e.g. Bosch Electronic Stability Control or lane-keeping assist) require control cycles of:

$$f_{\text{safety}} = 50\,\text{Hz} \implies T_{\text{deadline}} = 20.0\,\text{ms}$$

With our model executing at $0.18\,\text{ms}$:

$$\text{Timing Margin} = \frac{20.0\,\text{ms} - 0.18\,\text{ms}}{20.0\,\text{ms}} \times 100\% = 99.1\%$$

This leaves **$99.1\%$ of the ECU's execution time budget** available for other safety-critical software tasks (CAN-bus parsing, fault detection, sensor fusion, and actuator watchdogs).

---

## 8. Experimental Verification & Results

### 8.1 3D Closed-Loop Trajectory Verification

The model was subjected to a rigorous closed-loop trial over a complex $623.8\,\text{m}$ analytical track incorporating straightaways, sweeping turns, S-chicanes, and an urban $90^\circ$ intersection.

| Metric | Measured Result | Specification Target | Status |
| :--- | :---: | :---: | :---: |
| **Course Completion** | **$623.8\,\text{m} / 623.8\,\text{m}$ ($100\%$)** | $> 99\%$ | **PASSED** |
| **Maximum Lateral Tracking Error** | **$0.464\,\text{m}$** | $< 1.0\,\text{m}$ (Half-width: $3.75\,\text{m}$) | **PASSED** |
| **Mean Lateral Tracking Error** | **$0.142\,\text{m}$** | $< 0.3\,\text{m}$ | **PASSED** |
| **Cornering Deceleration Target** | **$28.5\,\text{km/h}$ achieved** | $\le 30\,\text{km/h}$ on $90^\circ$ turn | **PASSED** |
| **Destination Arrival Speed** | **$0.00\,\text{km/h}$** | $< 0.5\,\text{km/h}$ at gantry | **PASSED** |
| **Collisions / Off-Road Departures** | **0** | 0 | **PASSED** |

### 8.2 2D Multi-Lane Highway Verification

High-speed overtaking trials were conducted up to $200\,\text{km/h}$ in `simulate_highway_drive.py`:
- **Collision Rate:** $0.0\%$ across all tested scenarios ($100\,\text{km/h}, 150\,\text{km/h}, 200\,\text{km/h}$).
- **Lane Changing Efficiency:** Dynamic gap acceptance logic ensured minimum time-to-collision (TTC) remained $> 2.8\,\text{s}$ throughout overtaking maneuvers.

---

## 9. Conclusion & Phase 2 Roadmap

This project establishes a complete, verified, and deliverable Phase 1 autonomous vehicle control foundation. The system successfully demonstrates the full engineering path from raw sensor extraction to sub-millisecond edge neural network execution and interactive 3D physical simulation.

### Proposed Phase 2 Research Roadmap
1. **Temporal Memory Architectures (LSTM / Temporal Convolutional Networks):**
   Upgrading the feedforward controller to consume a rolling historical window ($T = 1.0\,\text{s}$, 10 frames) to learn dynamic vehicle inertia, tire slip angles, and higher-order jerk minimization.
2. **Multi-Agent 3D Traffic Simulation:**
   Integrating dynamic NPC traffic vehicles directly into the Ursina 3D simulation environment for live 3D obstacle avoidance, lane changing, and blind-spot monitoring.
3. **End-to-End Perception Fusion:**
   Coupling the motion controller directly with raw stereo camera vision embeddings and LiDAR range point clouds.

---
*Report generated automatically for project `KITTI_Vehicle_Motion_Analysis`.*
