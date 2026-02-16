use super::GaussianSplats3D;
use crate::components::{HalfSize3D, Translation3D};

impl GaussianSplats3D {
    /// Creates a new [`GaussianSplats3D`] with per-splat [`Self::centers`] and
    /// anisotropic [`Self::scales`].
    ///
    /// The scales represent per-axis standard deviations `(sx, sy, sz)` in **linear
    /// space**, forming the diagonal of the scale matrix used to reconstruct the 3D
    /// covariance: `Σ = R · diag(s²) · Rᵀ`.
    #[inline]
    pub fn from_centers_and_scales(
        centers: impl IntoIterator<Item = impl Into<Translation3D>>,
        scales: impl IntoIterator<Item = impl Into<HalfSize3D>>,
    ) -> Self {
        Self::new(centers, scales)
    }

}