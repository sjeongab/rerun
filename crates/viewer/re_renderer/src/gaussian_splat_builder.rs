//! Builder for Gaussian splat clouds, making it easy to create
//! [`crate::renderer::GaussianSplatDrawData`].

use itertools::Itertools as _;
use re_log::ResultExt as _;

use crate::allocator::DataTextureSource;
use crate::draw_phases::PickingLayerObjectId;
use crate::renderer::gpu_data::{PositionOpacity, Rotation, Scale};
use crate::renderer::{
    GaussianSplatBatchInfo, GaussianSplatDrawData, GaussianSplatDrawDataError,
};
use crate::{
    Color32, CpuWriteGpuReadError, DebugLabel, DepthOffset, OutlineMaskPreference,
    PickingLayerInstanceId, RenderContext,
};

/// Builder for Gaussian splat clouds, making it easy to create
/// [`crate::renderer::GaussianSplatDrawData`].
///
/// Splats **must** be added in back-to-front order (relative to the current
/// camera) for correct alpha compositing.
pub struct GaussianSplatBuilder<'ctx> {
    pub(crate) ctx: &'ctx RenderContext,

    // All buffers must stay the same length at all times.
    pub(crate) position_opacity_buffer: DataTextureSource<'ctx, PositionOpacity>,
    pub(crate) scale_buffer: DataTextureSource<'ctx, Scale>,
    pub(crate) rotation_buffer: DataTextureSource<'ctx, Rotation>,
    pub(crate) color_buffer: DataTextureSource<'ctx, Color32>,
    pub(crate) picking_instance_ids_buffer: DataTextureSource<'ctx, PickingLayerInstanceId>,

    pub(crate) batches: Vec<GaussianSplatBatchInfo>,

    pub(crate) radius_boost_in_ui_points_for_outlines: f32,
}

impl<'ctx> GaussianSplatBuilder<'ctx> {
    pub fn new(ctx: &'ctx RenderContext) -> Self {
        Self {
            ctx,
            position_opacity_buffer: DataTextureSource::new(ctx),
            scale_buffer: DataTextureSource::new(ctx),
            rotation_buffer: DataTextureSource::new(ctx),
            color_buffer: DataTextureSource::new(ctx),
            picking_instance_ids_buffer: DataTextureSource::new(ctx),
            batches: Vec::with_capacity(16),
            radius_boost_in_ui_points_for_outlines: 0.0,
        }
    }

    /// Returns the number of splats that can be added without reallocation.
    /// This may be smaller than the requested number if the maximum data texture
    /// size is reached.
    pub fn reserve(
        &mut self,
        expected_number_of_additional_splats: usize,
    ) -> Result<usize, CpuWriteGpuReadError> {
        // The maximum capacity is independent of datatype, so checking one is
        // sufficient.
        self.position_opacity_buffer
            .reserve(expected_number_of_additional_splats)?;
        self.scale_buffer
            .reserve(expected_number_of_additional_splats)?;
        self.rotation_buffer
            .reserve(expected_number_of_additional_splats)?;
        self.color_buffer
            .reserve(expected_number_of_additional_splats)?;
        self.picking_instance_ids_buffer
            .reserve(expected_number_of_additional_splats)
    }

    /// Boosts the size of splat quads by the given amount of ui-points for the
    /// purpose of drawing outlines.
    pub fn radius_boost_in_ui_points_for_outlines(
        &mut self,
        radius_boost_in_ui_points_for_outlines: f32,
    ) {
        self.radius_boost_in_ui_points_for_outlines = radius_boost_in_ui_points_for_outlines;
    }

    /// Start of a new batch.
    #[inline]
    pub fn batch(
        &mut self,
        label: impl Into<DebugLabel>,
    ) -> GaussianSplatBatchBuilder<'_, 'ctx> {
        self.batches.push(GaussianSplatBatchInfo {
            label: label.into(),
            ..GaussianSplatBatchInfo::default()
        });

        GaussianSplatBatchBuilder(self)
    }

    /// Start of a new batch with a fully specified [`GaussianSplatBatchInfo`].
    #[inline]
    pub fn batch_with_info(
        &mut self,
        info: GaussianSplatBatchInfo,
    ) -> GaussianSplatBatchBuilder<'_, 'ctx> {
        self.batches.push(info);

        GaussianSplatBatchBuilder(self)
    }

    /// Finalizes the builder and returns a draw data with all the splats added
    /// so far.
    pub fn into_draw_data(self) -> Result<GaussianSplatDrawData, GaussianSplatDrawDataError> {
        GaussianSplatDrawData::new(self)
    }
}

// ---------------------------------------------------------------------------
// Batch builder
// ---------------------------------------------------------------------------

pub struct GaussianSplatBatchBuilder<'a, 'ctx>(&'a mut GaussianSplatBuilder<'ctx>);

impl Drop for GaussianSplatBatchBuilder<'_, '_> {
    fn drop(&mut self) {
        // Remove the batch again if it wasn't actually used.
        if self.0.batches.last().unwrap().splat_count == 0 {
            self.0.batches.pop();
        }
    }
}

impl GaussianSplatBatchBuilder<'_, '_> {
    #[inline]
    fn batch_mut(&mut self) -> &mut GaussianSplatBatchInfo {
        self.0
            .batches
            .last_mut()
            .expect("batch should have been added on GaussianSplatBatchBuilder creation")
    }

    /// Sets the `world_from_obj` matrix for the *entire* batch.
    #[inline]
    pub fn world_from_obj(mut self, world_from_obj: glam::Affine3A) -> Self {
        self.batch_mut().world_from_obj = world_from_obj;
        self
    }

    /// Sets an outline mask for every element in the batch.
    #[inline]
    pub fn outline_mask_ids(mut self, outline_mask_ids: OutlineMaskPreference) -> Self {
        self.batch_mut().overall_outline_mask_ids = outline_mask_ids;
        self
    }

    /// Sets the depth offset for the entire batch.
    #[inline]
    pub fn depth_offset(mut self, depth_offset: DepthOffset) -> Self {
        self.batch_mut().depth_offset = depth_offset;
        self
    }

    /// Sets the picking object id for the current batch.
    #[inline]
    pub fn picking_object_id(mut self, picking_object_id: PickingLayerObjectId) -> Self {
        self.batch_mut().picking_object_id = picking_object_id;
        self
    }

    /// Pushes additional outline mask ids for a specific range of splats.
    /// The range is relative to this batch.
    ///
    /// Prefer the `outline_mask_ids` setting to set the outline mask for the
    /// entire batch whenever possible!
    #[inline]
    pub fn push_additional_outline_mask_ids_for_range(
        mut self,
        range: std::ops::Range<u32>,
        ids: OutlineMaskPreference,
    ) -> Self {
        self.batch_mut()
            .additional_outline_mask_ids_vertex_ranges
            .push((range, ids));
        self
    }

    /// Add several Gaussian splats to this batch.
    ///
    /// All slices should ideally have the same length. Missing values are
    /// padded with sensible defaults:
    /// - **opacity**: defaults to `1.0`
    /// - **scales**: defaults to `Vec3::ONE * 0.01`
    /// - **rotations**: defaults to `Quat::IDENTITY`
    /// - **colors**: defaults to `Color32::WHITE`
    /// - **picking_ids**: defaults to `PickingLayerInstanceId::default()`
    #[inline]
    pub fn add_splats(
        mut self,
        positions: &[glam::Vec3],
        opacities: &[f32],
        scales: &[glam::Vec3],
        rotations: &[glam::Quat],
        colors: &[Color32],
        picking_ids: &[PickingLayerInstanceId],
    ) -> Self {
        re_tracing::profile_function!();

        debug_assert_eq!(
            self.0.position_opacity_buffer.len(),
            self.0.scale_buffer.len()
        );
        debug_assert_eq!(
            self.0.position_opacity_buffer.len(),
            self.0.rotation_buffer.len()
        );
        debug_assert_eq!(
            self.0.position_opacity_buffer.len(),
            self.0.color_buffer.len()
        );
        debug_assert_eq!(
            self.0.position_opacity_buffer.len(),
            self.0.picking_instance_ids_buffer.len()
        );

        // Reserve ahead of time to check whether we're hitting the data
        // texture limit.
        let Some(num_available) = self
            .0
            .position_opacity_buffer
            .reserve(positions.len())
            .ok_or_log_error()
        else {
            return self;
        };

        let num_splats = if positions.len() > num_available {
            re_log::error_once!(
                "Reached maximum number of splats for Gaussian splat cloud of {}. \
                 Ignoring all excess splats.",
                self.0.position_opacity_buffer.len() + num_available
            );
            num_available
        } else {
            positions.len()
        };

        if num_splats == 0 {
            return self;
        }

        // Clamp input slices to the actual number of splats we'll add.
        let positions = &positions[..num_splats.min(positions.len())];
        let opacities = &opacities[..num_splats.min(opacities.len())];
        let scales = &scales[..num_splats.min(scales.len())];
        let rotations = &rotations[..num_splats.min(rotations.len())];
        let colors = &colors[..num_splats.min(colors.len())];
        let picking_ids = &picking_ids[..num_splats.min(picking_ids.len())];

        self.batch_mut().splat_count += num_splats as u32;

        // ---- Position + opacity ----
        {
            re_tracing::profile_scope!("position_opacity");

            let default_opacity = *opacities.last().unwrap_or(&1.0);
            let vertices: Vec<PositionOpacity> = positions
                .iter()
                .enumerate()
                .map(|(i, &pos)| PositionOpacity {
                    pos,
                    opacity: if i < opacities.len() {
                        opacities[i]
                    } else {
                        default_opacity
                    },
                })
                .collect();

            self.0
                .position_opacity_buffer
                .extend_from_slice(&vertices)
                .ok_or_log_error();
        }

        // ---- Scale ----
        {
            re_tracing::profile_scope!("scale");

            let default_scale = scales
                .last()
                .copied()
                .unwrap_or(glam::Vec3::splat(0.01));
            let vertices: Vec<Scale> = (0..num_splats)
                .map(|i| Scale {
                    scale: if i < scales.len() {
                        scales[i]
                    } else {
                        default_scale
                    },
                    _padding: 0.0,
                })
                .collect();

            self.0
                .scale_buffer
                .extend_from_slice(&vertices)
                .ok_or_log_error();
        }

        // ---- Rotation ----
        {
            re_tracing::profile_scope!("rotation");

            let default_rotation = rotations
                .last()
                .copied()
                .unwrap_or(glam::Quat::IDENTITY);
            let vertices: Vec<Rotation> = (0..num_splats)
                .map(|i| Rotation {
                    rotation: if i < rotations.len() {
                        rotations[i]
                    } else {
                        default_rotation
                    },
                })
                .collect();

            self.0
                .rotation_buffer
                .extend_from_slice(&vertices)
                .ok_or_log_error();
        }

        // ---- Color ----
        {
            re_tracing::profile_scope!("colors");

            self.0
                .color_buffer
                .extend_from_slice(colors)
                .ok_or_log_error();
            self.0
                .color_buffer
                .add_n(Color32::WHITE, num_splats.saturating_sub(colors.len()))
                .ok_or_log_error();
        }

        // ---- Picking instance ids ----
        {
            re_tracing::profile_scope!("picking_ids");

            self.0
                .picking_instance_ids_buffer
                .extend_from_slice(picking_ids)
                .ok_or_log_error();
            self.0
                .picking_instance_ids_buffer
                .add_n(
                    PickingLayerInstanceId::default(),
                    num_splats.saturating_sub(picking_ids.len()),
                )
                .ok_or_log_error();
        }

        self
    }
}