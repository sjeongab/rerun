import numpy as np
from plyfile import PlyData
import rerun as rr
from pathlib import Path

# --- CONFIGURATION ---
script_dir = Path(__file__).parent.resolve()
ply_file_path = script_dir / "ex_0.ply"
# ---------------------

SH_C0 = 0.28209479177387814


def load_splats():
    rr.init("gaussian_splat_viewer", spawn=True)

    print(f"Loading {ply_file_path}...")
    try:
        plydata = PlyData.read(str(ply_file_path))
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    v = plydata["vertex"]
    count = len(v["x"])
    print(f"Found {count} splats. Processing data...")

    # 1. Positions (means)
    positions = np.stack([v["x"], v["y"], v["z"]], axis=-1)

    # 2. Colors from SH DC band
    colors_sh = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1)
    colors = np.clip(0.5 + SH_C0 * colors_sh, 0.0, 1.0)
    # Convert to 0-255 RGBA for Rerun (alpha=255, opacity is separate)
    colors_u8 = (colors * 255).astype(np.uint8)

    # 3. Opacities — stored in logit-space, apply sigmoid
    opacities = 1.0 / (1.0 + np.exp(-v["opacity"]))

    # 4. Scales — stored in log-space, apply exp
    scales = np.exp(
        np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1)
    )

    # 5. Rotations — PLY stores (w, x, y, z), Rerun expects (x, y, z, w)
    quats = np.stack(
        [v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], axis=-1
    )
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norms

    print("Logging to Rerun...")

    mode = "ellipsoids"
    print(f"mode: {mode}")

    match mode.lower():
        case "points":
            avg_radius = np.mean(scales, axis=1)
            rgba = np.column_stack([colors_u8, (opacities * 255).astype(np.uint8)])
            rr.log(
                "points",
                rr.Points3D(
                    positions,
                    colors=rgba,
                    radii=avg_radius,
                ),
            )
        case "splats":
            rr.log(
                "splats",
                rr.GaussianSplats3D(
                    centers=positions,
                    scales=scales,
                    quaternions=quats,
                    opacities=opacities.astype(np.float32),
                    colors=colors_u8,
                ),
            )
        case "ellipsoids":
            rr.log(
                "ellipsoids",
                rr.Ellipsoids3D(
                    centers=positions,
                    half_sizes=scales,
                    quaternions=quats,
                    colors=np.column_stack(
                        [colors_u8, (opacities * 255).astype(np.uint8)]
                    ),
                    fill_mode="solid",
                ),
            )


if __name__ == "__main__":
    print(f"RERUN FILE: {rr.__file__}")
    print(f"RERUN VERSION: {rr.__version__}")

    if hasattr(rr, "GaussianSplats3D"):
        print("SUCCESS: GaussianSplats3D found!")
    else:
        print("FAILURE: GaussianSplats3D NOT found in this version.")
        print("Available attributes:", [x for x in dir(rr) if "3D" in x])

    load_splats()