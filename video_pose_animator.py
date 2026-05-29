"""
Video → Realistic 3D Animated Character
=========================================
Full pipeline: Real video → MediaPipe 3D Pose → Smoothing → 3D Mesh Animation → MP4/GIF

INSTALL:
    pip install opencv-python mediapipe matplotlib numpy scipy Pillow

USAGE:
    python video_pose_animator.py --video input.mp4
    python video_pose_animator.py --video input.mp4 --mode mesh   --output out.mp4
    python video_pose_animator.py --video input.mp4 --mode stick  --output out.gif
    python video_pose_animator.py --video input.mp4 --mode both   --fps 30

OUTPUTS:
    <name>_animation.mp4  — smooth 3D character animation video
    <name>_preview.gif    — quick GIF preview
    <name>_keypoints.npy  — raw extracted keypoints [N,33,3]

Author: Built on top of realistic_pose_3d.py mesh pipeline
"""

import cv2
import numpy as np
import math
import os
import sys
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import mediapipe as mp
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — JOINT / SKELETON DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# MediaPipe Pose landmark indices we care about
LANDMARK_IDX = {
    "nose":           0,
    "left_eye":       2,
    "right_eye":      5,
    "left_ear":       7,
    "right_ear":      8,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_heel":      29,
    "right_heel":     30,
    "left_foot":      31,
    "right_foot":     32,
}

# Skeleton connectivity for stick figure / edge rendering
# (joint_name_A, joint_name_B, color_hex, linewidth)
SKELETON_EDGES = [
    # Spine
    ("left_hip",      "right_hip",      "#FFD700", 4.0),
    ("left_shoulder", "right_shoulder", "#FFD700", 4.0),
    # Torso verticals
    ("left_shoulder", "left_hip",       "#FFA500", 3.5),
    ("right_shoulder","right_hip",      "#FFA500", 3.5),
    # Left arm
    ("left_shoulder", "left_elbow",     "#00BFFF", 3.0),
    ("left_elbow",    "left_wrist",     "#00BFFF", 2.5),
    # Right arm
    ("right_shoulder","right_elbow",    "#FF6347", 3.0),
    ("right_elbow",   "right_wrist",    "#FF6347", 2.5),
    # Left leg
    ("left_hip",      "left_knee",      "#7CFC00", 3.5),
    ("left_knee",     "left_ankle",     "#7CFC00", 3.0),
    ("left_ankle",    "left_foot",      "#7CFC00", 2.0),
    # Right leg
    ("right_hip",     "right_knee",     "#FF69B4", 3.5),
    ("right_knee",    "right_ankle",    "#FF69B4", 3.0),
    ("right_ankle",   "right_foot",     "#FF69B4", 2.0),
    # Head
    ("left_ear",      "right_ear",      "#DDA0DD", 3.0),
    ("left_ear",      "left_shoulder",  "#DDA0DD", 2.5),
    ("right_ear",     "right_shoulder", "#DDA0DD", 2.5),
    ("nose",          "left_eye",       "#FFFFFF", 2.0),
    ("nose",          "right_eye",      "#FFFFFF", 2.0),
]

# Body segment groups for mesh coloring
BODY_PARTS = {
    "torso":      ("#E8A87C", 1.0),   # skin / shirt
    "head":       ("#E8A87C", 1.0),   # skin
    "left_arm":   ("#4A90E2", 0.95),  # shirt color (left)
    "right_arm":  ("#E25C4A", 0.95),  # shirt color (right)
    "left_leg":   ("#2C3E7A", 0.95),  # pants left
    "right_leg":  ("#2C3E7A", 0.95),  # pants right
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VIDEO KEYPOINT EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class VideoKeypointExtractor:
    """
    Reads a video file and extracts MediaPipe 3D world landmarks
    for every frame. Handles low-confidence frames by interpolation.
    """

    def __init__(self, model_complexity: int = 2):
        self.mp_pose     = mp.solutions.pose
        self.mp_draw     = mp.solutions.drawing_utils
        self.complexity  = model_complexity

    def extract(self, video_path: str,
                max_frames: int = None,
                skip_frames: int = 1) -> dict:
        """
        Extract 3D pose from every frame.

        Returns dict:
            keypoints_3d : [N, 33, 3]  — world-space (meters, Y-up)
            keypoints_2d : [N, 33, 3]  — pixel (x,y) + depth z
            visibility   : [N, 33]     — per-landmark confidence
            timestamps   : [N]         — frame timestamps in seconds
            fps          : float
            resolution   : (W, H)
            n_detected   : int         — frames where pose was found
        """
        if not os.path.exists(video_path):
            sys.exit(f"[ERROR] Video not found: {video_path}")

        cap    = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if max_frames:
            total = min(total, max_frames)

        print(f"\n{'─'*55}")
        print(f"  Video : {video_path}")
        print(f"  Size  : {W}×{H} @ {fps:.1f} FPS  ({total} frames)")
        print(f"{'─'*55}")

        kps3d_all   = []
        kps2d_all   = []
        vis_all     = []
        timestamps  = []
        n_detected  = 0
        last_good   = None

        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        ) as pose:

            frame_idx = 0
            t0 = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and frame_idx >= max_frames:
                    break

                # Optional frame skipping
                if frame_idx % skip_frames != 0:
                    frame_idx += 1
                    continue

                ts  = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_world_landmarks and result.pose_landmarks:
                    # ── 3D world landmarks (metric, hip-centred) ──
                    wlm = result.pose_world_landmarks.landmark
                    kps3d = np.array(
                        [[l.x, -l.y, -l.z] for l in wlm],   # flip Y for 3D right-hand coords
                        dtype=np.float32)   # [33, 3]

                    # ── 2D image landmarks (normalised → pixel) ──
                    plm = result.pose_landmarks.landmark
                    kps2d = np.array(
                        [[l.x * W, l.y * H, l.z] for l in plm],
                        dtype=np.float32)  # [33, 3]

                    vis = np.array([l.visibility for l in wlm], dtype=np.float32)

                    kps3d_all.append(kps3d)
                    kps2d_all.append(kps2d)
                    vis_all.append(vis)
                    last_good = (kps3d.copy(), kps2d.copy(), vis.copy())
                    n_detected += 1

                else:
                    # Carry forward last known pose (better than zeros)
                    if last_good is not None:
                        kps3d_all.append(last_good[0].copy())
                        kps2d_all.append(last_good[1].copy())
                        vis_all.append(last_good[2].copy() * 0.5)  # lower confidence
                    else:
                        kps3d_all.append(np.zeros((33, 3), dtype=np.float32))
                        kps2d_all.append(np.zeros((33, 3), dtype=np.float32))
                        vis_all.append(np.zeros(33, dtype=np.float32))

                timestamps.append(ts)
                frame_idx += 1

                # Progress bar
                pct = frame_idx / total * 100
                bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                elapsed = time.time() - t0
                eta = elapsed / max(frame_idx, 1) * (total - frame_idx)
                print(f"\r  [{bar}] {pct:5.1f}%  frame {frame_idx}/{total}  "
                      f"ETA {eta:.0f}s  detected={n_detected}", end="", flush=True)

        cap.release()
        print(f"\n  Done! {n_detected}/{frame_idx} frames detected "
              f"({n_detected/max(frame_idx,1)*100:.1f}%)")

        return {
            "keypoints_3d": np.array(kps3d_all,  dtype=np.float32),  # [N, 33, 3]
            "keypoints_2d": np.array(kps2d_all,  dtype=np.float32),  # [N, 33, 3]
            "visibility":   np.array(vis_all,    dtype=np.float32),  # [N, 33]
            "timestamps":   np.array(timestamps, dtype=np.float32),  # [N]
            "fps":          fps,
            "resolution":   (W, H),
            "n_detected":   n_detected,
            "total_frames": len(kps3d_all),
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TEMPORAL SMOOTHER
# ══════════════════════════════════════════════════════════════════════════════

class TemporalSmoother:
    """
    Removes jitter from extracted keypoints using multiple methods.
    Savitzky-Golay is best for motion capture data.
    """

    @staticmethod
    def savgol(data: np.ndarray,
               window: int = 11,
               poly:   int = 3) -> np.ndarray:
        """
        Savitzky-Golay filter — best quality, preserves peaks.
        data: [N, 33, 3]
        """
        N, J, D = data.shape
        # Ensure window is odd and smaller than N
        window = min(window, N - (1 - N % 2))
        if window < poly + 2:
            window = poly + 2
        if window % 2 == 0:
            window += 1

        out = data.copy()
        for j in range(J):
            for d in range(D):
                try:
                    out[:, j, d] = savgol_filter(data[:, j, d], window, poly)
                except ValueError:
                    pass
        return out

    @staticmethod
    def gaussian(data: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Gaussian temporal blur — simple, slightly over-smooths peaks."""
        return gaussian_filter1d(data.astype(np.float64),
                                 sigma=sigma, axis=0).astype(np.float32)

    @staticmethod
    def ema(data: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Exponential moving average — causal / real-time capable."""
        out = data.copy()
        for i in range(1, len(data)):
            out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
        return out

    @classmethod
    def smooth(cls, data: np.ndarray,
               method: str = "savgol",
               **kw) -> np.ndarray:
        """Unified entry point."""
        print(f"\n  Smoothing {len(data)} frames with '{method}' filter...")
        if method == "savgol":
            out = cls.savgol(data, **kw)
        elif method == "gaussian":
            out = cls.gaussian(data, **kw)
        elif method == "ema":
            out = cls.ema(data, **kw)
        elif method == "combined":
            # Best quality: EMA first, then Savitzky-Golay
            out = cls.ema(data, alpha=0.6)
            out = cls.savgol(out, window=9, poly=3)
        else:
            out = data
        print(f"  Smoothing complete.")
        return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MESH GEOMETRY BUILDERS
# (adapted from realistic_pose_3d.py uploaded by user)
# ══════════════════════════════════════════════════════════════════════════════

def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v

def _local_frame(axis):
    axis = _norm(np.asarray(axis, float))
    ref  = np.array([0, 1, 0]) if abs(axis[1]) < 0.9 else np.array([1, 0, 0])
    u = _norm(np.cross(ref, axis))
    v = np.cross(axis, u)
    return u, v

def _cylinder_mesh(p0, p1, r0, r1=None, segs=12, bulge=1.0, bulge_t=0.4):
    """
    Tapered cylinder with optional belly bulge.
    r0 = radius at p0, r1 = radius at p1 (defaults to r0)
    bulge = max radius multiplier at bulge_t fraction along the segment.
    """
    if r1 is None:
        r1 = r0
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-8:
        return np.zeros((0, 3)), np.zeros((0, 3), int)

    u, v  = _local_frame(d)
    rings = 16
    th    = np.linspace(0, 2 * math.pi, segs, endpoint=False)
    circ  = np.array([math.cos(a) * u + math.sin(a) * v for a in th])

    verts = []
    for i in range(rings + 1):
        t = i / rings
        # radius profile: linear taper + gaussian bulge
        r_lin  = r0 + (r1 - r0) * t
        gauss  = bulge * math.exp(-((t - bulge_t) ** 2) / (2 * 0.12 ** 2))
        r      = r_lin * (1.0 + gauss * 0.22)
        pt     = p0 + d * t
        verts.append(pt + circ * r)

    verts  = np.vstack(verts)
    faces  = []
    for i in range(rings):
        for j in range(segs):
            k  = (j + 1) % segs
            a  = i * segs + j
            b  = i * segs + k
            c  = (i + 1) * segs + j
            dd = (i + 1) * segs + k
            faces += [[a, b, dd], [a, dd, c]]

    # caps
    bc = len(verts); verts_list = verts.tolist(); verts_list.append(p0.tolist())
    tc = bc + 1;    verts_list.append(p1.tolist())
    for j in range(segs):
        faces.append([bc, (j + 1) % segs, j])
        faces.append([tc, rings * segs + j, rings * segs + (j + 1) % segs])

    return np.array(verts_list, float), np.array(faces, int)

def _sphere_mesh(center, r, segs=10):
    """UV sphere."""
    center = np.asarray(center, float)
    stacks = segs
    verts  = []
    for i in range(stacks + 1):
        lat = math.pi * i / stacks - math.pi / 2
        for j in range(segs):
            lon = 2 * math.pi * j / segs
            verts.append([
                center[0] + r * math.cos(lat) * math.cos(lon),
                center[1] + r * math.sin(lat),
                center[2] + r * math.cos(lat) * math.sin(lon),
            ])
    verts = np.array(verts, float)
    faces = []
    for i in range(stacks):
        for j in range(segs):
            a = i * segs + j
            b = i * segs + (j + 1) % segs
            c = (i + 1) * segs + j
            d = (i + 1) * segs + (j + 1) % segs
            faces += [[a, b, d], [a, d, c]]
    return verts, np.array(faces, int)

def _ellipsoid_mesh(center, rx, ry, rz, segs=12):
    """Axis-aligned ellipsoid."""
    center = np.asarray(center, float)
    stacks = segs
    verts  = []
    for i in range(stacks + 1):
        lat = math.pi * i / stacks - math.pi / 2
        for j in range(segs):
            lon = 2 * math.pi * j / segs
            verts.append([
                center[0] + rx * math.cos(lat) * math.cos(lon),
                center[1] + ry * math.sin(lat),
                center[2] + rz * math.cos(lat) * math.sin(lon),
            ])
    verts = np.array(verts, float)
    faces = []
    for i in range(stacks):
        for j in range(segs):
            a = i * segs + j
            b = i * segs + (j + 1) % segs
            c = (i + 1) * segs + j
            d = (i + 1) * segs + (j + 1) % segs
            faces += [[a, b, d], [a, d, c]]
    return verts, np.array(faces, int)


def build_body_mesh_for_frame(J: dict) -> list:
    """
    Build a list of (vertices, faces, color, alpha) tuples — one per body part.
    J: dict of {joint_name: np.array([x,y,z])}

    Returns separate part meshes so each can be coloured differently.
    """
    sho_w = np.linalg.norm(J["left_shoulder"] - J["right_shoulder"]) + 1e-6
    hip_w = np.linalg.norm(J["left_hip"]      - J["right_hip"])      + 1e-6

    parts = []  # list of (verts, faces, rgba)

    # ── Skin / shirt color palette ────────────────────────────────────────────
    SKIN   = (0.88, 0.70, 0.56, 0.97)
    SHIRT  = (1.00, 0.55, 0.10, 0.95)
    SHORTS = (0.96, 0.96, 0.96, 0.95)
    SOCKS  = (0.96, 0.96, 0.96, 0.95)
    SHOES  = (0.95, 0.95, 0.95, 0.97)
    HAIR   = (0.22, 0.14, 0.08, 0.98)

    def add(v, f, c):
        if len(v) > 0:
            parts.append((v, f, c))

    # ── HEAD ──────────────────────────────────────────────────────────────────
    ear_mid  = (J["left_ear"] + J["right_ear"]) / 2
    head_ctr = ear_mid + np.array([0, sho_w * 0.10, 0])
    head_r   = sho_w * 0.195
    head_ry  = head_r * 1.18
    v, f = _ellipsoid_mesh(head_ctr, head_r, head_ry, head_r * 0.95, segs=14)
    add(v, f, SKIN)

    # Hair cap (upper hemisphere)
    hair_ctr = head_ctr + np.array([0, head_ry * 0.15, 0])
    v, f = _ellipsoid_mesh(hair_ctr, head_r * 0.98, head_ry * 0.62, head_r * 0.93, segs=12)
    add(v, f, HAIR)

    # ── NECK ──────────────────────────────────────────────────────────────────
    neck_bot = (J["left_shoulder"] + J["right_shoulder"]) / 2
    neck_top = head_ctr - np.array([0, head_ry * 0.82, 0])
    v, f = _cylinder_mesh(neck_bot, neck_top, sho_w * 0.09, sho_w * 0.085, segs=10)
    add(v, f, SKIN)

    # ── TORSO (shaped: wider shoulders, narrower waist, hip flare) ────────────
    mid_sho = (J["left_shoulder"] + J["right_shoulder"]) / 2
    mid_hip = (J["left_hip"]      + J["right_hip"])      / 2
    spine   = mid_sho - mid_hip
    spine_L = np.linalg.norm(spine)

    # Left torso side
    v, f = _cylinder_mesh(J["left_hip"], J["left_shoulder"],
                           hip_w * 0.30, sho_w * 0.28,
                           segs=12, bulge=0.18, bulge_t=0.45)
    add(v, f, SHIRT)

    # Right torso side
    v, f = _cylinder_mesh(J["right_hip"], J["right_shoulder"],
                           hip_w * 0.30, sho_w * 0.28,
                           segs=12, bulge=0.18, bulge_t=0.45)
    add(v, f, SHIRT)

    # Centre spine fill
    v, f = _cylinder_mesh(mid_hip, mid_sho,
                           (hip_w + sho_w) / 4 * 0.72, None,
                           segs=14, bulge=0.08)
    add(v, f, SHIRT)

    # Hip cross-bar
    v, f = _cylinder_mesh(J["left_hip"], J["right_hip"],
                           hip_w * 0.19, None, segs=10)
    add(v, f, SHORTS)

    # Shoulder cross-bar
    v, f = _cylinder_mesh(J["left_shoulder"], J["right_shoulder"],
                           sho_w * 0.16, None, segs=10)
    add(v, f, SHIRT)

    # ── ARMS ─────────────────────────────────────────────────────────────────
    arm_colors = {
        "left":  (SHIRT[:3] + (0.95,)),
        "right": (SHIRT[:3] + (0.95,)),
    }
    for side in ("left", "right"):
        sho   = J[f"{side}_shoulder"]
        elbow = J[f"{side}_elbow"]
        wrist = J[f"{side}_wrist"]
        c     = arm_colors[side]

        # Shoulder cap
        v, f = _sphere_mesh(sho, sho_w * 0.115, segs=9)
        add(v, f, SHIRT)

        # Upper arm (bicep/tricep shape)
        v, f = _cylinder_mesh(sho, elbow, sho_w * 0.106, sho_w * 0.09,
                               segs=12, bulge=0.28, bulge_t=0.40)
        add(v, f, SKIN)

        # Elbow sphere
        v, f = _sphere_mesh(elbow, sho_w * 0.088, segs=8)
        add(v, f, SKIN)

        # Forearm (taper toward wrist)
        v, f = _cylinder_mesh(elbow, wrist, sho_w * 0.088, sho_w * 0.072,
                               segs=10, bulge=0.20, bulge_t=0.30)
        add(v, f, SKIN)

        # Hand (small ellipsoid)
        hand_mid = wrist + (wrist - elbow) * 0.25
        v, f = _ellipsoid_mesh(hand_mid, sho_w * 0.068, sho_w * 0.055,
                                sho_w * 0.030, segs=8)
        add(v, f, SKIN)

    # ── LEGS ──────────────────────────────────────────────────────────────────
    for side in ("left", "right"):
        hip   = J[f"{side}_hip"]
        knee  = J[f"{side}_knee"]
        ankle = J[f"{side}_ankle"]
        foot  = J[f"{side}_foot"]

        # Hip sphere
        v, f = _sphere_mesh(hip, hip_w * 0.215, segs=10)
        add(v, f, SHORTS)

        # Thigh (quad muscle shape)
        v, f = _cylinder_mesh(hip, knee, hip_w * 0.238, hip_w * 0.195,
                               segs=14, bulge=0.30, bulge_t=0.35)
        add(v, f, SHORTS)

        # Knee sphere
        v, f = _sphere_mesh(knee, hip_w * 0.162, segs=9)
        add(v, f, SKIN)

        # Shin / calf
        v, f = _cylinder_mesh(knee, ankle, hip_w * 0.165, hip_w * 0.115,
                               segs=12, bulge=0.28, bulge_t=0.30)
        add(v, f, SOCKS)

        # Ankle sphere
        v, f = _sphere_mesh(ankle, hip_w * 0.110, segs=8)
        add(v, f, SOCKS)

        # Foot (flattened ellipsoid pointing forward)
        fwd   = foot - ankle
        fwd_L = np.linalg.norm(fwd) + 1e-6
        fwd   = fwd / fwd_L
        foot_ctr = ankle + fwd * fwd_L * 0.55
        v, f = _ellipsoid_mesh(foot_ctr,
                                fwd_L * 0.55,
                                hip_w * 0.085,
                                hip_w * 0.130,
                                segs=10)
        add(v, f, SHOES)

    return parts  # [(verts, faces, rgba), ...]


def jdict_from_row(kps: np.ndarray) -> dict:
    """Convert [33, 3] array → joint name dict."""
    J = {}
    for name, idx in LANDMARK_IDX.items():
        J[name] = kps[idx]
    return J


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — STICK FIGURE RENDERER
# ══════════════════════════════════════════════════════════════════════════════

class StickFigureRenderer:
    """
    Renders a glowing neon stick figure on a dark background.
    Fast — suitable for real-time preview.
    """

    def render_frame(self, ax, kps: np.ndarray, frame_idx: int = 0,
                      total_frames: int = 1):
        """Draw one frame's stick figure onto ax."""
        J = jdict_from_row(kps)

        ax.set_facecolor("#0D0D1A")

        for (jA, jB, col, lw) in SKELETON_EDGES:
            if jA not in J or jB not in J:
                continue
            pA, pB = J[jA], J[jB]
            ax.plot([pA[0], pB[0]], [pA[2], pB[2]], [pA[1], pB[1]],
                    color=col, linewidth=lw, alpha=0.92, solid_capstyle='round')

        # Joint dots
        for name, idx in LANDMARK_IDX.items():
            p = kps[idx]
            ax.scatter(p[0], p[2], p[1], color="white", s=14, zorder=5, alpha=0.8)

        # Progress bar (right side)
        pct = frame_idx / max(total_frames - 1, 1)
        ax.text2D(0.97, 0.05, f"▶ {frame_idx+1}/{total_frames}",
                  transform=ax.transAxes, color="#888888",
                  fontsize=7, ha="right")
        ax.text2D(0.97, 0.01, "─" * int(40 * pct),
                  transform=ax.transAxes, color="#FFD700",
                  fontsize=5, ha="right")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MESH RENDERER
# ══════════════════════════════════════════════════════════════════════════════

class MeshRenderer:
    """
    Renders a realistic body mesh using Poly3DCollection in matplotlib.
    Slower than stick figure but looks much better.
    """

    def render_frame(self, ax, kps: np.ndarray, frame_idx: int = 0,
                      total_frames: int = 1):
        """Build and draw mesh parts for one frame."""
        J = jdict_from_row(kps)
        ax.set_facecolor("#111118")

        parts = build_body_mesh_for_frame(J)

        for (verts, faces, rgba) in parts:
            if len(verts) == 0:
                continue

            # Build triangle list for Poly3DCollection
            tris = []
            for face in faces:
                if max(face) < len(verts):
                    tri = [
                        [verts[face[0], 0], verts[face[0], 2], verts[face[0], 1]],
                        [verts[face[1], 0], verts[face[1], 2], verts[face[1], 1]],
                        [verts[face[2], 0], verts[face[2], 2], verts[face[2], 1]],
                    ]
                    tris.append(tri)

            if not tris:
                continue

            poly = Poly3DCollection(tris, alpha=rgba[3])
            poly.set_facecolor(rgba[:3])
            poly.set_edgecolor("none")
            ax.add_collection3d(poly)

        ax.text2D(0.97, 0.05, f"▶ {frame_idx+1}/{total_frames}",
                  transform=ax.transAxes, color="#888888",
                  fontsize=7, ha="right")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ANIMATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AnimationEngine:
    """
    Builds and exports the full 3D animation from smoothed keypoints.
    """

    def __init__(self,
                 keypoints_3d: np.ndarray,
                 fps: float = 30.0,
                 mode: str = "stick",
                 dpi: int = 120,
                 figsize: tuple = (9, 9)):
        """
        keypoints_3d : [N, 33, 3]
        mode         : 'stick' | 'mesh' | 'both'
        """
        self.kps       = keypoints_3d
        self.fps       = fps
        self.mode      = mode
        self.dpi       = dpi
        self.figsize   = figsize
        self.N         = len(keypoints_3d)

        # Compute scene bounds across ALL frames for stable camera
        self.bounds = self._compute_bounds()

        self.stick_renderer = StickFigureRenderer()
        self.mesh_renderer  = MeshRenderer()

    def _compute_bounds(self) -> dict:
        """Compute stable XYZ bounds across all frames."""
        all_pts = self.kps.reshape(-1, 3)
        pad = 0.20
        return {
            "x": (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad),
            "y": (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad),
            "z": (all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad),
        }

    def _setup_ax(self, ax, title="", elev=12, azim=-70):
        """Apply consistent axes styling."""
        b = self.bounds
        ax.set_xlim(b["x"]); ax.set_ylim(b["z"]); ax.set_zlim(b["y"])
        ax.set_xlabel("X", color="#444", fontsize=8)
        ax.set_ylabel("Z", color="#444", fontsize=8)
        ax.set_zlabel("Y", color="#444", fontsize=8)
        ax.tick_params(colors="#333", labelsize=6)
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.12, color="#555")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#1A1A2A")
        ax.yaxis.pane.set_edgecolor("#1A1A2A")
        ax.zaxis.pane.set_edgecolor("#1A1A2A")
        if title:
            ax.set_title(title, color="#CCCCCC", fontsize=9, pad=4)

    def _make_frame_stick(self, frame_idx: int) -> np.ndarray:
        """Render one stick figure frame → numpy RGBA image."""
        fig = plt.figure(figsize=self.figsize, facecolor="#0D0D1A")
        ax  = fig.add_subplot(111, projection='3d',
                               facecolor="#0D0D1A")
        self._setup_ax(ax, title=f"3D Pose — frame {frame_idx+1}/{self.N}")
        self.stick_renderer.render_frame(ax, self.kps[frame_idx],
                                          frame_idx, self.N)
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(h, w, 3)
        plt.close(fig)
        return img

    def _make_frame_mesh(self, frame_idx: int) -> np.ndarray:
        """Render one mesh frame → numpy RGB image."""
        fig = plt.figure(figsize=self.figsize, facecolor="#111118")
        ax  = fig.add_subplot(111, projection='3d',
                               facecolor="#111118")
        self._setup_ax(ax, title=f"3D Human — frame {frame_idx+1}/{self.N}")
        self.mesh_renderer.render_frame(ax, self.kps[frame_idx],
                                         frame_idx, self.N)
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(h, w, 3)
        plt.close(fig)
        return img

    def _make_frame_both(self, frame_idx: int) -> np.ndarray:
        """Side-by-side: stick left, mesh right."""
        fig = plt.figure(figsize=(self.figsize[0] * 2, self.figsize[1]),
                          facecolor="#0A0A14")
        ax1 = fig.add_subplot(121, projection='3d', facecolor="#0D0D1A")
        ax2 = fig.add_subplot(122, projection='3d', facecolor="#111118")

        self._setup_ax(ax1, title="Stick Figure", azim=-70)
        self._setup_ax(ax2, title="Mesh Body", azim=-55)

        self.stick_renderer.render_frame(ax1, self.kps[frame_idx],
                                          frame_idx, self.N)
        self.mesh_renderer.render_frame(ax2, self.kps[frame_idx],
                                         frame_idx, self.N)

        fig.suptitle(f"3D Human Pose Animation  •  frame {frame_idx+1}/{self.N}",
                     color="#DDDDDD", fontsize=11, y=0.99)
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(h, w, 3)
        plt.close(fig)
        return img

    def render_all_frames(self, mode: str = None) -> list:
        """
        Render every frame to a numpy image.
        Returns list of [H, W, 3] uint8 arrays.
        """
        mode = mode or self.mode
        render_fn = {
            "stick": self._make_frame_stick,
            "mesh":  self._make_frame_mesh,
            "both":  self._make_frame_both,
        }.get(mode, self._make_frame_stick)

        frames = []
        t0 = time.time()
        for i in range(self.N):
            img = render_fn(i)
            frames.append(img)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (self.N - i - 1)
            pct = (i + 1) / self.N * 100
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"\r  Rendering [{bar}] {pct:5.1f}%  frame {i+1}/{self.N}  "
                  f"ETA {eta:.0f}s", end="", flush=True)

        print(f"\n  Rendered {self.N} frames in {time.time()-t0:.1f}s")
        return frames

    def export_mp4(self, frames: list, output_path: str):
        """Write frames to MP4 using OpenCV VideoWriter."""
        if not frames:
            return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
        for frame in frames:
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        vw.release()
        sz = os.path.getsize(output_path) / 1024 / 1024
        print(f"  MP4 → {output_path}  ({sz:.1f} MB,  {len(frames)} frames @ {self.fps:.0f} FPS)")

    def export_gif(self, frames: list, output_path: str,
                   max_frames: int = 90,
                   scale: float = 0.5):
        """Write frames to animated GIF (downsampled for file size)."""
        if not frames:
            return
        # Subsample to max_frames
        step = max(1, len(frames) // max_frames)
        sel  = frames[::step][:max_frames]
        delay_ms = int(1000 / self.fps * step)

        pil_frames = []
        for f in sel:
            h, w = f.shape[:2]
            nh, nw = int(h * scale), int(w * scale)
            resized = cv2.resize(f, (nw, nh))
            pil_frames.append(Image.fromarray(resized).convert("P", palette=Image.ADAPTIVE))

        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=delay_ms,
            optimize=True,
        )
        sz = os.path.getsize(output_path) / 1024 / 1024
        print(f"  GIF → {output_path}  ({sz:.1f} MB,  {len(pil_frames)} frames)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — QUALITY METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_quality_metrics(raw: np.ndarray, smoothed: np.ndarray) -> dict:
    """Compute jitter reduction and smoothness score."""
    def jitter(x):
        if len(x) < 3:
            return 0.0
        d2 = np.diff(x, n=2, axis=0)
        return float(np.mean(np.abs(d2)))

    raw_j = jitter(raw)
    smo_j = jitter(smoothed)
    reduction = (raw_j - smo_j) / (raw_j + 1e-8) * 100.0
    smoothness = 1.0 - min(smo_j / (raw_j + 1e-8), 1.0)

    return {
        "jitter_raw":       raw_j,
        "jitter_smoothed":  smo_j,
        "jitter_reduction": reduction,
        "smoothness_score": smoothness,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(args):
    base     = os.path.splitext(os.path.basename(args.video))[0]
    out_dir  = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    mp4_path = os.path.join(out_dir, f"{base}_animation.mp4")
    gif_path = os.path.join(out_dir, f"{base}_preview.gif")
    npy_path = os.path.join(out_dir, f"{base}_keypoints.npy")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║   VIDEO → 3D ANIMATED CHARACTER                         ║
╠══════════════════════════════════════════════════════════╣
║   Input  : {args.video:<44} ║
║   Mode   : {args.mode:<44} ║
║   Smooth : {args.smooth:<44} ║
║   Output : {out_dir:<44} ║
╚══════════════════════════════════════════════════════════╝""")

    # ── STEP 1: Extract keypoints ─────────────────────────────────────────────
    print("\n[1/5] Extracting 3D keypoints from video...")
    extractor = VideoKeypointExtractor(model_complexity=args.complexity)
    data      = extractor.extract(
        args.video,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
    )

    kps_raw = data["keypoints_3d"]   # [N, 33, 3]
    fps     = data["fps"]
    N       = data["total_frames"]

    if args.fps_override:
        fps = args.fps_override

    print(f"  Extracted: {N} frames  |  FPS: {fps:.1f}  |  "
          f"Detection: {data['n_detected']/N*100:.1f}%")

    # ── Save raw keypoints ────────────────────────────────────────────────────
    np.save(npy_path, kps_raw)
    print(f"  Keypoints saved → {npy_path}")

    # ── STEP 2: Smooth keypoints ──────────────────────────────────────────────
    print(f"\n[2/5] Smoothing with '{args.smooth}' filter...")
    smoother  = TemporalSmoother()
    kps_smooth = smoother.smooth(kps_raw, method=args.smooth,
                                  window=args.sg_window, poly=args.sg_poly)

    metrics = compute_quality_metrics(kps_raw, kps_smooth)
    print(f"  Jitter reduction : {metrics['jitter_reduction']:.1f}%")
    print(f"  Smoothness score : {metrics['smoothness_score']:.3f}")

    # ── STEP 3: Build animation engine ────────────────────────────────────────
    print(f"\n[3/5] Initialising animation engine (mode={args.mode})...")
    engine = AnimationEngine(
        keypoints_3d=kps_smooth,
        fps=fps,
        mode=args.mode,
        dpi=args.dpi,
        figsize=(args.figsize, args.figsize),
    )

    # ── STEP 4: Render frames ─────────────────────────────────────────────────
    print(f"\n[4/5] Rendering {N} frames (this takes a while)...")
    frames = engine.render_all_frames()

    # ── STEP 5: Export ────────────────────────────────────────────────────────
    print(f"\n[5/5] Exporting...")
    engine.export_mp4(frames, mp4_path)
    if args.gif:
        engine.export_gif(frames, gif_path,
                          max_frames=args.gif_frames,
                          scale=args.gif_scale)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════╗
║   COMPLETE                                               ║
╠══════════════════════════════════════════════════════════╣
║   Frames    : {N:<43} ║
║   Duration  : {N/fps:.1f}s @ {fps:.0f} FPS{'':<35} ║
║   Jitter ↓  : {metrics['jitter_reduction']:.1f}%{'':<42} ║
║   MP4       : {os.path.basename(mp4_path):<43} ║""")
    if args.gif:
        print(f"║   GIF       : {os.path.basename(gif_path):<43} ║")
    print("╚══════════════════════════════════════════════════════════╝")

    return mp4_path, gif_path


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Video → Realistic 3D Animated Character",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python video_pose_animator.py --video person.mp4
  python video_pose_animator.py --video dance.mp4 --mode both --gif
  python video_pose_animator.py --video walk.mp4  --mode mesh --fps 24
  python video_pose_animator.py --video squat.mp4 --smooth combined --max-frames 150
        """
    )
    # I/O
    p.add_argument("--video",       required=True,
                   help="Input video file (mp4, avi, mov, mkv, ...)")
    p.add_argument("--output-dir",  default="output_animation",
                   help="Output directory [default: output_animation/]")

    # Mode
    p.add_argument("--mode",        choices=["stick", "mesh", "both"],
                   default="stick",
                   help="Render mode: stick=fast, mesh=realistic, both=side-by-side")

    # Smoothing
    p.add_argument("--smooth",      choices=["savgol", "gaussian", "ema",
                                              "combined", "none"],
                   default="combined",
                   help="Temporal smoothing algorithm [default: combined]")
    p.add_argument("--sg-window",   type=int, default=11,
                   help="Savitzky-Golay window size (odd) [default: 11]")
    p.add_argument("--sg-poly",     type=int, default=3,
                   help="Savitzky-Golay polynomial order [default: 3]")

    # Render quality
    p.add_argument("--dpi",         type=int,   default=100,
                   help="Render DPI [default: 100]")
    p.add_argument("--figsize",     type=float, default=7.5,
                   help="Figure size in inches [default: 7.5]")
    p.add_argument("--fps",         type=float, default=None,
                   dest="fps_override",
                   help="Override output FPS [default: match input]")
    p.add_argument("--complexity",  type=int,   default=2, choices=[0, 1, 2],
                   help="MediaPipe model complexity 0-2 [default: 2]")

    # Frame control
    p.add_argument("--max-frames",  type=int, default=None,
                   help="Limit frames processed (for quick tests)")
    p.add_argument("--skip-frames", type=int, default=1,
                   help="Process every Nth frame [default: 1 = all frames]")

    # GIF export
    p.add_argument("--gif",         action="store_true",
                   help="Also export animated GIF preview")
    p.add_argument("--gif-frames",  type=int,   default=60,
                   help="Max frames in GIF [default: 60]")
    p.add_argument("--gif-scale",   type=float, default=0.45,
                   help="GIF resolution scale [default: 0.45]")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
