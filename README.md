# 🧍 Human Pose to 3D Mesh Pipeline

Convert a **single image** or **video** into a realistic posed 3D human mesh and animation — no training required.

---

## Overview

This project contains two complementary scripts that form a full image/video → 3D character pipeline:

| Script | Input | Output |
|---|---|---|
| `realistic_pose_3d.py` | Single image (`.jpg`, `.png`) | Posed `.obj` 3D mesh + `.mtl` material + skeleton preview |
| `video_pose_animator.py` | Video (`.mp4`, `.avi`, `.mov`) | Smoothed 3D animation as `.mp4` and/or `.gif` |

Both scripts use **MediaPipe** for markerless pose estimation and build a realistic human mesh from scratch — with muscle-shaped limbs, anatomically correct torso, head, hands, and feet.

---

## Features

- **Zero training** — runs inference on any image or video out of the box
- **Realistic mesh geometry** — muscle-belly cylinders with bulge/taper, joint spheres, shaped torso, hands, and feet
- **SMPL body model support** — optionally uses SMPL for higher-fidelity meshes when model weights are available
- **Temporal smoothing** — Savitzky-Golay, Gaussian, EMA, and combined filters to eliminate jitter from video
- **Multiple render modes** — stick figure, mesh, or side-by-side
- **Blender-ready output** — auto-generated `blender_smooth.py` script for subdivision and smoothing on import
- **Color sampling** — samples skin, shirt, pants, and shoe colors directly from the source image for material generation

---

## Installation

```bash
pip install mediapipe opencv-python numpy scipy matplotlib Pillow
```

**Optional — for SMPL mesh (higher quality):**
```bash
pip install smplx torch
```
Then download `SMPL_NEUTRAL.pkl` from [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) and place it in `smpl_models/`.

---

## Usage

### Image → 3D Mesh

```bash
python realistic_pose_3d.py --image person.jpg
```

**Outputs:**
- `person_posed.obj` — posed 3D human mesh
- `person_posed.mtl` — material with sampled colors
- `person_preview.jpg` — skeleton overlay on original image
- `blender_smooth.py` — Blender import + subdivision script

```bash
# Custom output path
python realistic_pose_3d.py --image person.jpg --output results/my_mesh.obj

# Use custom SMPL model directory
python realistic_pose_3d.py --image person.jpg --smpl path/to/smpl_models/
```

---

### Video → 3D Animation

```bash
python video_pose_animator.py --video input.mp4
```

**Common options:**

```bash
# Mesh render mode + export GIF
python video_pose_animator.py --video dance.mp4 --mode mesh --gif

# Side-by-side stick + mesh
python video_pose_animator.py --video walk.mp4 --mode both --fps 24

# Quick test with frame limit
python video_pose_animator.py --video squat.mp4 --max-frames 100

# Custom smoothing
python video_pose_animator.py --video run.mp4 --smooth savgol --sg-window 15
```

**Render modes:**

| Mode | Description |
|---|---|
| `stick` | Fast colored skeleton render |
| `mesh` | Realistic 3D body mesh |
| `both` | Side-by-side comparison |

**Smoothing algorithms:**

| Method | Description |
|---|---|
| `combined` | Savitzky-Golay + Gaussian (default, best quality) |
| `savgol` | Savitzky-Golay filter |
| `gaussian` | Gaussian blur over time |
| `ema` | Exponential moving average |
| `none` | Raw keypoints |

**Outputs:**
- `<name>_animation.mp4` — rendered 3D animation
- `<name>_preview.gif` — lightweight GIF preview
- `<name>_keypoints.npy` — raw extracted keypoints `[N, 33, 3]`

---

## Blender Import

After running `realistic_pose_3d.py`, import the mesh into Blender:

1. `File > Import > Wavefront OBJ` → select the generated `.obj`
2. Open the **Scripting** tab
3. Run `blender_smooth.py` — applies smooth shading, subdivision surface (level 2), and corrective smooth

---

## Pipeline Architecture

```
Image / Video
     │
     ▼
MediaPipe Pose Estimation
(33 world-space landmarks)
     │
     ├─── [Video only] Temporal Smoothing
     │         Savitzky-Golay / Gaussian / EMA
     │
     ▼
3D Mesh Construction
  ├── Head (ellipsoid + facial geometry)
  ├── Neck (tapered cylinder)
  ├── Torso (8-ring cross-section with hip/waist/chest shaping)
  ├── Arms (muscle cylinders + joint spheres + hands)
  └── Legs (quad/calf bulge cylinders + feet)
     │
     ├─── [SMPL available] SMPL body model override
     │
     ▼
Export
  ├── .obj + .mtl (image pipeline)
  └── .mp4 + .gif + .npy (video pipeline)
```

---

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- NumPy
- SciPy *(video pipeline only)*
- Matplotlib *(video pipeline only)*
- Pillow *(video pipeline only)*
- `smplx` + `torch` *(optional, for SMPL mesh)*

---

## Tech Stack

`MediaPipe` · `OpenCV` · `NumPy` · `SciPy` · `Matplotlib` · `SMPL` (optional) · `Blender` (optional post-processing)

---

## License

MIT
