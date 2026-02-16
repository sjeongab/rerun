// crates/viewer/re_view_spatial/src/visualizers/gaussian_splats3d.rs

use itertools::Itertools as _;
use re_renderer::{GaussianSplatBuilder, PickingLayerInstanceId};
use re_sdk_types::archetypes::GaussianSplats3D;
use re_sdk_types::components::{Color, HalfSize3D, Opacity, Position3D, RotationQuat};
use re_view::{process_annotation_and_keypoint_slices, process_color_slice};
use re_viewer_context::{
    IdentifiedViewSystem, QueryContext, ViewContext, ViewContextCollection, ViewQuery,
    ViewSystemExecutionError, VisualizerExecutionOutput, VisualizerQueryInfo, VisualizerSystem,
};

use super::SpatialViewVisualizerData;
use crate::contexts::SpatialSceneVisualizerInstructionContext;
use crate::view_kind::SpatialViewKind;

// ---------------------------------------------------------------------------

/// Cutoff in standard deviations for the bounding radius of each splat.
const SIGMA_CUTOFF: f32 = 3.0;

// ---------------------------------------------------------------------------

pub struct GaussianSplats3DVisualizer {
    pub data: SpatialViewVisualizerData,
}

impl Default for GaussianSplats3DVisualizer {
    fn default() -> Self {
        Self {
            data: SpatialViewVisualizerData::new(Some(SpatialViewKind::ThreeD)),
        }
    }
}

// ---------------------------------------------------------------------------

struct GaussianSplats3DComponentData<'a> {
    // Required
    centers: &'a [Position3D],

    // Clamped to edge
    scales: &'a [HalfSize3D],
    quaternions: &'a [RotationQuat],
    colors: &'a [Color],
    opacities: &'a [f32],
}

// ---------------------------------------------------------------------------

impl GaussianSplats3DVisualizer {
    fn process_data<'a>(
        &mut self,
        ctx: &QueryContext<'_>,
        splat_builder: &mut GaussianSplatBuilder<'_>,
        query: &ViewQuery<'_>,
        ent_context: &SpatialSceneVisualizerInstructionContext<'_>,
        data: impl Iterator<Item = GaussianSplats3DComponentData<'a>>,
    ) -> Result<(), ViewSystemExecutionError> {
        let entity_path = ctx.target_entity_path;

        for data in data {
            let num_instances = data.centers.len();
            if num_instances == 0 {
                continue;
            }

            let picking_ids = (0..num_instances)
                .map(|i| PickingLayerInstanceId(i as _))
                .collect_vec();

            // No keypoints or class ids for splats — just resolve annotation colors.
            let (annotation_infos, _keypoints) = process_annotation_and_keypoint_slices(
                query.latest_at,
                num_instances,
                data.centers.iter().map(|p| p.0.into()),
                &[],
                &[],
                &ent_context.annotations,
            );

            let positions: &[glam::Vec3] = bytemuck::cast_slice(data.centers);

            let obj_space_bounding_box =
                macaw::BoundingBox::from_points(positions.iter().copied());

            // Extract scales as glam::Vec3 slices.
            let scales: &[glam::Vec3] = if data.scales.is_empty() {
                &[]
            } else {
                bytemuck::cast_slice(data.scales)
            };

            // Extract rotations as glam::Quat slices.
            let rotations: &[glam::Quat] = if data.quaternions.is_empty() {
                &[]
            } else {
                bytemuck::cast_slice(data.quaternions)
            };

            // Resolve colors through the annotation pipeline.
            let colors = process_color_slice(
                ctx,
                GaussianSplats3D::descriptor_colors().component,
                num_instances,
                &annotation_infos,
                data.colors,
            );

            // Opacities are passed directly to the builder — the shader
            // handles per-splat opacity via the position_opacity data texture.
            let opacities: &[f32] = data.opacities;

            // Submit for each transform instance (mirrors Points3D).
            for world_from_obj in ent_context
                .transform_info
                .target_from_instances()
                .iter()
                .map(|transform| transform.as_affine3a())
            {
                let splat_batch = splat_builder
                    .batch(entity_path.to_string())
                    .world_from_obj(world_from_obj)
                    .outline_mask_ids(ent_context.highlight.overall)
                    .picking_object_id(re_renderer::PickingLayerObjectId(
                        entity_path.hash64(),
                    ));

                let mut splat_range_builder = splat_batch.add_splats(
                    positions,
                    opacities,
                    scales,
                    rotations,
                    &colors,
                    &picking_ids,
                );

                // Per-instance highlight outlines.
                for (highlighted_key, instance_mask_ids) in &ent_context.highlight.instances {
                    let idx = highlighted_key.get();
                    if idx < num_instances as u64 {
                        splat_range_builder = splat_range_builder
                            .push_additional_outline_mask_ids_for_range(
                                idx as u32..idx as u32 + 1,
                                *instance_mask_ids,
                            );
                    }
                }

                // Compute bounding box using scales for a conservative extent.
                // Expand each position by its largest scale axis × SIGMA_CUTOFF.
                {
                    let mut expanded_bbox = obj_space_bounding_box;
                    if !scales.is_empty() {
                        for (i, &pos) in positions.iter().enumerate() {
                            let s = scales[i.min(scales.len() - 1)];
                            let max_s = s.x.abs().max(s.y.abs()).max(s.z.abs());
                            let extent = glam::Vec3::splat(max_s * SIGMA_CUTOFF);
                            expanded_bbox.extend(pos - extent);
                            expanded_bbox.extend(pos + extent);
                        }
                    }

                    self.data.add_bounding_box(
                        entity_path.hash(),
                        expanded_bbox,
                        world_from_obj,
                    );
                }
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------

impl IdentifiedViewSystem for GaussianSplats3DVisualizer {
    fn identifier() -> re_viewer_context::ViewSystemIdentifier {
        "GaussianSplats3D".into()
    }
}

impl VisualizerSystem for GaussianSplats3DVisualizer {
    fn visualizer_query_info(
        &self,
        _app_options: &re_viewer_context::AppOptions,
    ) -> VisualizerQueryInfo {
        VisualizerQueryInfo::from_archetype::<GaussianSplats3D>()
    }

    fn execute(
        &mut self,
        ctx: &ViewContext<'_>,
        view_query: &ViewQuery<'_>,
        context_systems: &ViewContextCollection,
    ) -> Result<VisualizerExecutionOutput, ViewSystemExecutionError> {
        let output = VisualizerExecutionOutput::default();

        let mut splat_builder = GaussianSplatBuilder::new(ctx.viewer_ctx.render_ctx());
        splat_builder.radius_boost_in_ui_points_for_outlines(
            re_view::SIZE_BOOST_IN_POINTS_FOR_POINT_OUTLINES,
        );

        use super::entity_iterator::process_archetype;
        process_archetype::<Self, GaussianSplats3D, _>(
            ctx,
            view_query,
            context_systems,
            &output,
            self.data.preferred_view_kind,
            |ctx, spatial_ctx, results| {
                let all_centers =
                    results.iter_required(GaussianSplats3D::descriptor_centers().component);
                if all_centers.is_empty() {
                    return Ok(());
                }

                let num_splats: usize = all_centers
                    .chunks()
                    .iter()
                    .flat_map(|chunk| chunk.iter_slices::<[f32; 3]>())
                    .map(|pts| pts.len())
                    .sum();
                if num_splats == 0 {
                    return Ok(());
                }

                splat_builder.reserve(num_splats)?;

                let all_scales =
                    results.iter_optional(GaussianSplats3D::descriptor_scales().component);
                let all_quaternions =
                    results.iter_optional(GaussianSplats3D::descriptor_quaternions().component);
                let all_opacities =
                    results.iter_optional(GaussianSplats3D::descriptor_opacities().component);
                let all_colors =
                    results.iter_optional(GaussianSplats3D::descriptor_colors().component);
                let data = re_query::range_zip_1x4(
                    all_centers.slice::<[f32; 3]>(),
                    all_scales.slice::<[f32; 3]>(),
                    all_quaternions.slice::<[f32; 4]>(),
                    all_opacities.slice::<f32>(),
                    all_colors.slice::<u32>(),
                )
                .map(
                    |(_index, centers, scales, quaternions, opacities, colors)| {
                        GaussianSplats3DComponentData {
                            centers: bytemuck::cast_slice(centers),
                            scales: scales.map_or(&[], bytemuck::cast_slice),
                            quaternions: quaternions.map_or(&[], bytemuck::cast_slice),
                            opacities: opacities.unwrap_or(&[]),
                            colors: colors.map_or(&[], |c| bytemuck::cast_slice(c)),
                        }
                    },
                );

                self.process_data(ctx, &mut splat_builder, view_query, spatial_ctx, data)
            },
        )?;

        Ok(output.with_draw_data([splat_builder.into_draw_data()?.into()]))
    }

    fn data(&self) -> Option<&dyn std::any::Any> {
        Some(self.data.as_any())
    }
}