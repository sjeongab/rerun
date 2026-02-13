import numpy as np
from plyfile import PlyData
import sys
import os
import rerun as rr
from pathlib import Path

# --- CONFIGURATION ---
script_dir = Path(__file__).parent.resolve()

# 2. Create the full path to the ply file
ply_file_path = script_dir / "point_cloud.ply"

CAMERA_POSITION = np.array([0.0, 0.0, -5.0])
# ---------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def load_splats():
    rr.init("gaussian_splat_viewer", spawn=True)

    print(f"Loading {ply_file_path}...")
    try:
        plydata = PlyData.read(str(ply_file_path))
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    v = plydata['vertex']
    count = len(v['x'])
    print(f"Found {count} splats. Processing data...")

    # 1. Positions (Means)
    positions = np.stack([v['x'], v['y'], v['z']], axis=-1)

    # 2. Colors (Spherical Harmonics DC)
    # Standard Splats store color as 'f_dc_0', 'f_dc_1', 'f_dc_2'
    # We need to convert SH to RGB: 0.5 + C0 * f_dc
    SH_C0 = 0.28209479177387814
    colors = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1)
    colors = 0.5 + (SH_C0 * colors)
    colors = np.clip(colors, 0, 1) # Ensure valid RGB range

    # 3. Opacity
    # Stored as 'opacity' in log-space (logit). Need sigmoid to get 0-1.
    opacities = 1.0 / (1.0 + np.exp(-v['opacity']))

    rgba_colors = np.column_stack([colors, opacities])
    # 4. Scales
    # Stored as 'scale_0', etc. in log-space. Need exp to get actual size.
    scales = np.exp(np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1))
    avg_radius = np.mean(scales, axis=1)


    diffs = positions - CAMERA_POSITION
    dists_sq = np.sum(diffs**2, axis=1)
    sort_indices = np.argsort(dists_sq)[::-1]

    sorted_pos = positions[sort_indices]
    sorted_colors = colors[sort_indices]
    sorted_opacities = opacities[sort_indices]
    sorted_radii = avg_radius[sort_indices]

    rgba_colors = np.column_stack([sorted_colors, sorted_opacities])
    # 5. Rotations (Quaternions)
    # Stored as 'rot_0' (w), 'rot_1' (x), 'rot_2' (y), 'rot_3' (z)
    # Rerun expects (x, y, z, w) order usually, but let's load them as is.
    # We must normalize them.
    quats = np.stack([v['rot_1'], v['rot_2'], v['rot_3'], v['rot_0']], axis=-1)
    # Normalize quaternions (x^2 + y^2 + z^2 + w^2 = 1)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norms

    print("Logging to Rerun...")

    # Log as actual Gaussian Splats
    rr.log(
        "splats",
        rr.GaussianSplats3D(
            sorted_pos,
            colors=rgba_colors,
            radii=sorted_radii,
        )
    )

if __name__ == "__main__":

    print(f"RERUN FILE: {rr.__file__}")
    print(f"RERUN VERSION: {rr.__version__}")

    # Check if the class exists in the loaded module
    if hasattr(rr, 'GaussianSplats3D'):
        print("SUCCESS: GaussianSplats3D found!")
    else:
        print("FAILURE: GaussianSplats3D NOT found in this version.")
        # Print what IS available to help debug
        print("Available attributes:", [x for x in dir(rr) if '3D' in x])
    load_splats()
