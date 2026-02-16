# gaussian_splats3d_ext.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..error_utils import catch_and_log_exceptions

if TYPE_CHECKING:
    from .. import datatypes


class GaussianSplats3DExt:
    """Extension for [GaussianSplats3D][rerun.archetypes.GaussianSplats3D]."""

    def __init__(
        self: Any,
        centers: datatypes.Vec3DArrayLike,
        scales: datatypes.Vec3DArrayLike,
        *,
        quaternions: datatypes.QuaternionArrayLike | None = None,
        opacities: datatypes.Float32ArrayLike | None = None,
        colors: datatypes.Rgba32ArrayLike | None = None,
    ) -> None:
        """
        Create a new instance of the GaussianSplats3D archetype.

        Parameters
        ----------
        centers:
            Per-splat 3D center positions in object (local) space.
        scales:
            Per-splat anisotropic scale (standard deviation) along each of the three
            local axes, in **linear space**.

            These correspond directly to the diagonal of the scale matrix
            `S = diag(sx, sy, sz)` used when reconstructing the 3D covariance:
            `Σ = R · S² · Rᵀ`.

            If your training framework outputs log-scales, apply `exp()` before logging.
        quaternions:
            Per-splat rotation as a unit quaternion in `(x, y, z, w)` convention.

            Defines the orientation of the splat's local frame relative to the object
            frame.  If omitted, all splats are axis-aligned (identity rotation).
        opacities:
            Per-splat base opacity in the range [0, 1].

            This is the peak alpha at the Gaussian center before the spatial falloff is
            applied.  If your source data stores opacity in logit-space, apply
            `sigmoid()` before logging.

            Defaults to `1.0` if not specified.
        colors:
            Per-splat color in linear RGBA.

            The RGB channels define the base color of the Gaussian.  The alpha channel,
            if present, acts as an additional visualization-level transparency multiplier
            applied on top of `opacities`.

            Defaults to white `(1, 1, 1, 1)` if not specified.

        """

        with catch_and_log_exceptions(context=self.__class__.__name__):
            # Normalize opacities: clamp to [0, 1] if provided.
            if opacities is not None:
                opacities = np.clip(
                    np.asarray(opacities, dtype=np.float32), 0.0, 1.0
                )

            self.__attrs_init__(
                centers=centers,
                scales=scales,
                quaternions=quaternions,
                opacities=opacities,
                colors=colors,
            )
            return

        self.__attrs_clear__()