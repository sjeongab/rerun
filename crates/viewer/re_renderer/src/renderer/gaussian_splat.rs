//! Gaussian Splat renderer for efficient rendering of 3D Gaussian splat clouds.
//!
//! How it works:
//! =================
//! Each Gaussian splat is rendered as a camera-facing quad. The vertex shader
//! reads per-splat data from data textures (position+opacity, scale, rotation,
//! color), projects the 3D covariance into screen space, and expands a quad
//! that covers the 2D Gaussian footprint. The fragment shader evaluates the
//! 2D Gaussian for smooth alpha falloff.
//!
//! Like `super::point_cloud::PointCloudRenderer`, all quads are rendered in a
//! single triangle-list draw call per batch with no vertex buffers — all data
//! is fetched from data textures in the vertex shader.
//!
//! For WebGL compatibility, data is uploaded as textures.  Color is stored in a
//! separate sRGB texture so that sRGB→linear conversion happens on texture load.
//!
//! **Sorting**: Splats must be provided in **back-to-front** order for correct
//! alpha compositing.  The `GaussianSplatBuilder` is responsible for depth-sorting
//! before handing data to [`GaussianSplatDrawData::new`].
//!
//! Bind-group layout (must match `gaussian_splat.wgsl`):
//! ```text
//! group(0)            — global bindings (camera, etc.)  [re_renderer provided]
//! group(1) binding(0) — t_position_opacity  texture_2d<f32>  Rgba32Float
//! group(1) binding(1) — t_scale             texture_2d<f32>  Rgba32Float
//! group(1) binding(2) — t_rotation          texture_2d<f32>  Rgba32Float
//! group(1) binding(3) — t_color             texture_2d<f32>  Rgba8UnormSrgb
//! group(1) binding(4) — t_picking_ids       texture_2d<u32>  Rg32Uint
//! group(1) binding(5) — draw_data           uniform          DrawDataUniformBuffer
//! group(2) binding(0) — batch               uniform          BatchUniformBuffer
//! ```

use std::num::NonZeroU64;
use std::ops::Range;

use enumset::{EnumSet, enum_set};
use itertools::Itertools as _;
use smallvec::smallvec;

use super::{DrawData, DrawError, RenderContext, Renderer};
use crate::allocator::create_and_fill_uniform_buffer_batch;
use crate::draw_phases::{
    DrawPhase, OutlineMaskProcessor, PickingLayerObjectId, PickingLayerProcessor,
};
use crate::renderer::{DrawDataDrawable, DrawInstruction, DrawableCollectionViewInfo};
use crate::view_builder::ViewBuilder;
use crate::wgpu_resources::{
    BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, GpuBindGroup, GpuBindGroupLayoutHandle,
    GpuRenderPipelineHandle, GpuRenderPipelinePoolAccessor, PipelineLayoutDesc, RenderPipelineDesc,
};
use crate::{
    DebugLabel, DepthOffset, DrawableCollector, OutlineMaskPreference, GaussianSplatBuilder,
    include_shader_module,
};

// ---------------------------------------------------------------------------
// GPU data — must stay in sync with `gaussian_splat.wgsl`
// ---------------------------------------------------------------------------

pub mod gpu_data {
    use crate::draw_phases::PickingLayerObjectId;
    use crate::wgpu_buffer_types;

    // Per-splat data lives in data textures, not buffers, so standard buffer
    // alignment rules do *not* apply.  We use `#[repr(C, packed)]` to match
    // the texel layout exactly.

    /// One texel per splat in the position/opacity texture (Rgba32Float).
    ///
    /// WGSL: `textureLoad(t_position_opacity, texel_coord, 0)` → `vec4<f32>`
    ///       .xyz = position (object space), .w = opacity [0,1].
    #[repr(C, packed)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct PositionOpacity {
        pub pos: glam::Vec3,
        pub opacity: f32,
    }
    static_assertions::assert_eq_size!(PositionOpacity, glam::Vec4);

    /// One texel per splat in the scale texture (Rgba32Float).
    ///
    /// WGSL: `textureLoad(t_scale, texel_coord, 0).xyz` → per-axis std-dev.
    #[repr(C, packed)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct Scale {
        pub scale: glam::Vec3,
        pub _padding: f32,
    }
    static_assertions::assert_eq_size!(Scale, glam::Vec4);

    /// One texel per splat in the rotation texture (Rgba32Float).
    ///
    /// WGSL: `textureLoad(t_rotation, texel_coord, 0)` → quaternion (x,y,z,w).
    #[repr(C, packed)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct Rotation {
        pub rotation: glam::Quat,
    }
    static_assertions::assert_eq_size!(Rotation, glam::Vec4);

    // Color uses the same `[u8; 4]` / `Color32` texel as the point-cloud
    // renderer, written into an `Rgba8UnormSrgb` texture.
    //
    // Picking instance ids use the same `[u32; 2]` layout as the point-cloud
    // renderer, written into an `Rg32Uint` texture.

    /// Uniform buffer that changes once per draw-data (group 1, binding 5).
    ///
    /// Must be 256 bytes (16 rows × 16 bytes) for WebGPU compat.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct DrawDataUniformBuffer {
        /// Extra screen-space radius added when rendering outline masks.
        pub radius_boost_in_ui_points: wgpu_buffer_types::F32RowPadded,
        pub end_padding: [wgpu_buffer_types::PaddingRow; 16 - 1],
    }

    /// Uniform buffer that changes per batch (group 2, binding 0).
    ///
    /// Must be 256 bytes (16 rows × 16 bytes) for WebGPU compat.
    ///
    /// Layout in WGSL (each row is `vec4<f32>` / 16 bytes):
    /// ```text
    /// rows 0-3 : world_from_obj (mat4x4<f32>)
    /// row  4   : (depth_offset, 0, 0, 0)
    /// row  5   : (outline_mask_ids.xy, picking_object_id.xy)
    /// rows 6-15: padding
    /// ```
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct BatchUniformBuffer {
        pub world_from_obj: wgpu_buffer_types::Mat4,

        pub depth_offset: f32,
        pub _row_padding: [f32; 3],

        pub outline_mask_ids: wgpu_buffer_types::UVec2,
        pub picking_object_id: PickingLayerObjectId,

        pub end_padding: [wgpu_buffer_types::PaddingRow; 16 - 6],
    }
}

// ---------------------------------------------------------------------------
// Public batch description
// ---------------------------------------------------------------------------

/// Configuration for one batch of Gaussian splats sharing a transform.
pub struct GaussianSplatBatchInfo {
    pub label: DebugLabel,

    /// Object→world transform applied to every splat in the batch.
    pub world_from_obj: glam::Affine3A,

    /// Number of splats covered by this batch.
    ///
    /// Batches are consecutive: the first batch starts at splat 0, and each
    /// subsequent batch starts where the previous one ended.
    pub splat_count: u32,

    /// Optional outline mask for the entire batch.
    pub overall_outline_mask_ids: OutlineMaskPreference,

    /// Per-vertex-range outline mask overrides (ranges are batch-relative).
    pub additional_outline_mask_ids_vertex_ranges: Vec<(Range<u32>, OutlineMaskPreference)>,

    /// Picking object id for the entire batch.
    pub picking_object_id: PickingLayerObjectId,

    /// Depth offset applied after projection.
    pub depth_offset: DepthOffset,
}

impl Default for GaussianSplatBatchInfo {
    #[inline]
    fn default() -> Self {
        Self {
            label: DebugLabel::default(),
            world_from_obj: glam::Affine3A::IDENTITY,
            splat_count: 0,
            overall_outline_mask_ids: OutlineMaskPreference::NONE,
            additional_outline_mask_ids_vertex_ranges: Vec::new(),
            picking_object_id: Default::default(),
            depth_offset: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum GaussianSplatDrawDataError {
    #[error("Failed to transfer data to the GPU: {0}")]
    FailedTransferringDataToGpu(#[from] crate::allocator::CpuWriteGpuReadError),
}

// ---------------------------------------------------------------------------
// Internal batch (ready to draw)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct GaussianSplatBatch {
    bind_group: GpuBindGroup,
    vertex_range: Range<u32>,
    active_phases: EnumSet<DrawPhase>,
}

// ---------------------------------------------------------------------------
// Draw data
// ---------------------------------------------------------------------------

/// A Gaussian-splat drawing operation.  Expected to be recreated every frame.
#[derive(Clone)]
pub struct GaussianSplatDrawData {
    bind_group_all_splats: Option<GpuBindGroup>,
    bind_group_all_splats_outline_mask: Option<GpuBindGroup>,
    batches: Vec<GaussianSplatBatch>,
}

impl DrawData for GaussianSplatDrawData {
    type Renderer = GaussianSplatRenderer;

    fn collect_drawables(
        &self,
        _view_info: &DrawableCollectionViewInfo,
        collector: &mut DrawableCollector<'_>,
    ) {
        // TODO(sorting): Gaussian splats are transparent — ideally they would
        // be submitted to a dedicated transparent phase and sorted by batch
        // distance.  For now we draw them last in the opaque phase.
        for (batch_idx, batch) in self.batches.iter().enumerate() {
            collector.add_drawable(
                batch.active_phases,
                DrawDataDrawable {
                    distance_sort_key: f32::MAX,
                    draw_data_payload: batch_idx as _,
                },
            );
        }
    }
}

impl GaussianSplatDrawData {
    /// Transforms and uploads Gaussian-splat data to be consumed by the GPU.
    ///
    /// The builder must provide splat data in **back-to-front sorted order**
    /// relative to the current camera for correct alpha compositing.
    ///
    /// If no batches are present in the builder, all splats are treated as a
    /// single batch with identity transform.
    pub fn new(builder: GaussianSplatBuilder<'_>) -> Result<Self, GaussianSplatDrawDataError> {
        re_tracing::profile_function!();

        let GaussianSplatBuilder {
            ctx,
            position_opacity_buffer,
            scale_buffer,
            rotation_buffer,
            color_buffer,
            picking_instance_ids_buffer,
            batches,
            radius_boost_in_ui_points_for_outlines,
        } = builder;

        let renderer = ctx.renderer::<GaussianSplatRenderer>();
        let batches = batches.as_slice();

        if position_opacity_buffer.is_empty() {
            return Ok(Self {
                bind_group_all_splats: None,
                bind_group_all_splats_outline_mask: None,
                batches: Vec::new(),
            });
        }

        let num_splats = position_opacity_buffer.len();

        let fallback_batches = [GaussianSplatBatchInfo {
            label: "fallback_batch".into(),
            world_from_obj: glam::Affine3A::IDENTITY,
            splat_count: num_splats as _,
            overall_outline_mask_ids: OutlineMaskPreference::NONE,
            additional_outline_mask_ids_vertex_ranges: Vec::new(),
            picking_object_id: Default::default(),
            depth_offset: 0,
        }];
        let batches = if batches.is_empty() {
            &fallback_batches
        } else {
            batches
        };

        // ---- Finish data textures ----

        let position_opacity_texture = position_opacity_buffer.finish(
            wgpu::TextureFormat::Rgba32Float,
            "GaussianSplatDrawData::position_opacity_texture",
        )?;
        let scale_texture = scale_buffer.finish(
            wgpu::TextureFormat::Rgba32Float,
            "GaussianSplatDrawData::scale_texture",
        )?;
        let rotation_texture = rotation_buffer.finish(
            wgpu::TextureFormat::Rgba32Float,
            "GaussianSplatDrawData::rotation_texture",
        )?;
        let color_texture = color_buffer.finish(
            wgpu::TextureFormat::Rgba8UnormSrgb,
            "GaussianSplatDrawData::color_texture",
        )?;
        let picking_instance_id_texture = picking_instance_ids_buffer.finish(
            wgpu::TextureFormat::Rg32Uint,
            "GaussianSplatDrawData::picking_instance_id_texture",
        )?;

        // ---- Per-draw-data uniform buffers (normal + outline boost) ----

        let draw_data_uniform_buffer_bindings = create_and_fill_uniform_buffer_batch(
            ctx,
            "GaussianSplatDrawData::DrawDataUniformBuffer".into(),
            [
                gpu_data::DrawDataUniformBuffer {
                    radius_boost_in_ui_points: 0.0.into(),
                    end_padding: Default::default(),
                },
                gpu_data::DrawDataUniformBuffer {
                    radius_boost_in_ui_points: radius_boost_in_ui_points_for_outlines.into(),
                    end_padding: Default::default(),
                },
            ]
            .into_iter(),
        );
        let (draw_data_uniform_buffer_normal, draw_data_uniform_buffer_outline) =
            draw_data_uniform_buffer_bindings
                .into_iter()
                .collect_tuple()
                .unwrap();

        // ---- Bind groups for all-splats data (group 1) ----
        // Entry order must match the bind-group-layout binding order.

        let mk_bind_group = |label, draw_data_uniform_buffer_binding| {
            ctx.gpu_resources.bind_groups.alloc(
                &ctx.device,
                &ctx.gpu_resources,
                &BindGroupDesc {
                    label,
                    entries: smallvec![
                        BindGroupEntry::DefaultTextureView(position_opacity_texture.handle), // binding 0
                        BindGroupEntry::DefaultTextureView(scale_texture.handle),             // binding 1
                        BindGroupEntry::DefaultTextureView(rotation_texture.handle),          // binding 2
                        BindGroupEntry::DefaultTextureView(color_texture.handle),             // binding 3
                        BindGroupEntry::DefaultTextureView(picking_instance_id_texture.handle), // binding 4
                        draw_data_uniform_buffer_binding,                                      // binding 5
                    ],
                    layout: renderer.bind_group_layout_all_splats,
                },
            )
        };

        let bind_group_all_splats = mk_bind_group(
            "GaussianSplatDrawData::bind_group_all_splats".into(),
            draw_data_uniform_buffer_normal,
        );
        let bind_group_all_splats_outline_mask = mk_bind_group(
            "GaussianSplatDrawData::bind_group_all_splats_outline_mask".into(),
            draw_data_uniform_buffer_outline,
        );

        // ---- Per-batch uniform buffers & internal batch objects ----

        let mut batches_internal = Vec::with_capacity(batches.len());
        {
            let uniform_buffer_bindings = create_and_fill_uniform_buffer_batch(
                ctx,
                "gaussian splat batch uniform buffers".into(),
                batches
                    .iter()
                    .map(|batch_info| gpu_data::BatchUniformBuffer {
                        world_from_obj: batch_info.world_from_obj.into(),
                        depth_offset: batch_info.depth_offset as f32,
                        _row_padding: [0.0; 3],
                        outline_mask_ids: batch_info
                            .overall_outline_mask_ids
                            .0
                            .unwrap_or_default()
                            .into(),
                        picking_object_id: batch_info.picking_object_id,
                        end_padding: Default::default(),
                    }),
            );

            // Additional "micro batches" for per-range outline masks.
            let mut uniform_buffer_bindings_mask_only_batches =
                create_and_fill_uniform_buffer_batch(
                    ctx,
                    "gaussian splat batch uniform buffers - mask only".into(),
                    batches
                        .iter()
                        .flat_map(|batch_info| {
                            batch_info
                                .additional_outline_mask_ids_vertex_ranges
                                .iter()
                                .map(|(_, mask)| gpu_data::BatchUniformBuffer {
                                    world_from_obj: batch_info.world_from_obj.into(),
                                    depth_offset: batch_info.depth_offset as f32,
                                    _row_padding: [0.0; 3],
                                    outline_mask_ids: mask.0.unwrap_or_default().into(),
                                    picking_object_id: batch_info.picking_object_id,
                                    end_padding: Default::default(),
                                })
                        })
                        .collect::<Vec<_>>()
                        .into_iter(),
                )
                .into_iter();

            let mut start_splat_for_next_batch = 0u32;
            for (batch_info, uniform_buffer_binding) in
                batches.iter().zip(uniform_buffer_bindings.into_iter())
            {
                let splat_range_end = start_splat_for_next_batch + batch_info.splat_count;

                let mut active_phases = enum_set![DrawPhase::Transparent | DrawPhase::PickingLayer];
                if batch_info.overall_outline_mask_ids.is_some() {
                    active_phases.insert(DrawPhase::OutlineMask);
                }

                batches_internal.push(renderer.create_gaussian_splat_batch(
                    ctx,
                    batch_info.label.clone(),
                    uniform_buffer_binding,
                    start_splat_for_next_batch..splat_range_end,
                    active_phases,
                ));

                for (range, _) in &batch_info.additional_outline_mask_ids_vertex_ranges {
                    let range = (range.start + start_splat_for_next_batch)
                        ..(range.end + start_splat_for_next_batch);
                    batches_internal.push(renderer.create_gaussian_splat_batch(
                        ctx,
                        format!("{:?} mask-only {:?}", batch_info.label, range).into(),
                        uniform_buffer_bindings_mask_only_batches.next().unwrap(),
                        range.clone(),
                        enum_set![DrawPhase::OutlineMask],
                    ));
                }

                start_splat_for_next_batch = splat_range_end;

                // Clamp if vertex count was capped.
                if start_splat_for_next_batch >= num_splats as u32 {
                    break;
                }
            }
        }

        Ok(Self {
            bind_group_all_splats: Some(bind_group_all_splats),
            bind_group_all_splats_outline_mask: Some(bind_group_all_splats_outline_mask),
            batches: batches_internal,
        })
    }
}

// ---------------------------------------------------------------------------
// Renderer
// ---------------------------------------------------------------------------

pub struct GaussianSplatRenderer {
    render_pipeline_color: GpuRenderPipelineHandle,
    render_pipeline_picking_layer: GpuRenderPipelineHandle,
    render_pipeline_outline_mask: GpuRenderPipelineHandle,
    bind_group_layout_all_splats: GpuBindGroupLayoutHandle,
    bind_group_layout_batch: GpuBindGroupLayoutHandle,
}

impl GaussianSplatRenderer {
    fn create_gaussian_splat_batch(
        &self,
        ctx: &RenderContext,
        label: DebugLabel,
        uniform_buffer_binding: BindGroupEntry,
        splat_range: Range<u32>,
        active_phases: EnumSet<DrawPhase>,
    ) -> GaussianSplatBatch {
        let bind_group = ctx.gpu_resources.bind_groups.alloc(
            &ctx.device,
            &ctx.gpu_resources,
            &BindGroupDesc {
                label,
                entries: smallvec![uniform_buffer_binding],
                layout: self.bind_group_layout_batch,
            },
        );

        GaussianSplatBatch {
            bind_group,
            // 6 vertices per splat-quad (2 triangles), same convention as point_cloud.
            vertex_range: (splat_range.start * 6)..(splat_range.end * 6),
            active_phases,
        }
    }
}

impl Renderer for GaussianSplatRenderer {
    type RendererDrawData = GaussianSplatDrawData;

    fn create_renderer(ctx: &RenderContext) -> Self {
        re_tracing::profile_function!();

        let render_pipelines = &ctx.gpu_resources.render_pipelines;

        // ---- Bind-group layout: all splats (group 1) ----
        let bind_group_layout_all_splats = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &BindGroupLayoutDesc {
                label: "GaussianSplatRenderer::bind_group_layout_all_splats".into(),
                entries: vec![
                    // binding 0 — position + opacity (Rgba32Float)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1 — scale (Rgba32Float)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 2 — rotation quaternion (Rgba32Float)
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 3 — color (Rgba8UnormSrgb, read as float after sRGB decode)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 4 — picking instance ids (Rg32Uint)
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 5 — draw-data uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(std::mem::size_of::<
                                gpu_data::DrawDataUniformBuffer,
                            >(
                            )
                                as _),
                        },
                        count: None,
                    },
                ],
            },
        );

        // ---- Bind-group layout: per-batch (group 2) ----
        let bind_group_layout_batch = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &BindGroupLayoutDesc {
                label: "GaussianSplatRenderer::bind_group_layout_batch".into(),
                entries: vec![wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(std::mem::size_of::<
                            gpu_data::BatchUniformBuffer,
                        >() as _),
                    },
                    count: None,
                }],
            },
        );

        // ---- Pipeline layout ----
        let pipeline_layout = ctx.gpu_resources.pipeline_layouts.get_or_create(
            ctx,
            &PipelineLayoutDesc {
                label: "GaussianSplatRenderer::pipeline_layout".into(),
                entries: vec![
                    ctx.global_bindings.layout, // group 0
                    bind_group_layout_all_splats, // group 1
                    bind_group_layout_batch,      // group 2
                ],
            },
        );

        // ---- Shader module ----
        let shader_module_desc = include_shader_module!("../../shader/gaussian_splat.wgsl");
        let shader_module = ctx
            .gpu_resources
            .shader_modules
            .get_or_create(ctx, &shader_module_desc);

        // ---- Color pipeline (premultiplied-alpha blend, depth-write OFF) ----
        let render_pipeline_desc_color = RenderPipelineDesc {
            label: "GaussianSplatRenderer::render_pipeline_color".into(),
            pipeline_layout,
            vertex_entrypoint: "vs_main".into(),
            vertex_handle: shader_module,
            fragment_entrypoint: "fs_main".into(),
            fragment_handle: shader_module,
            vertex_buffers: smallvec![],
            render_targets: smallvec![Some(wgpu::ColorTargetState {
                format: ViewBuilder::MAIN_TARGET_COLOR_FORMAT,
                blend: Some(wgpu::BlendState {
                    // Pre-multiplied alpha: out = src + (1 − src.a) × dst
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: ViewBuilder::MAIN_TARGET_DEPTH_FORMAT,
                // Splats are transparent — we read depth but never write it.
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Greater, // reversed-Z
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: ViewBuilder::main_target_default_msaa_state(
                ctx.render_config(),
                false, // no alpha-to-coverage — we use real alpha blending
            ),
        };
        let render_pipeline_color =
            render_pipelines.get_or_create(ctx, &render_pipeline_desc_color);

        // ---- Picking-layer pipeline ----
        let render_pipeline_picking_layer = render_pipelines.get_or_create(
            ctx,
            &RenderPipelineDesc {
                label: "GaussianSplatRenderer::render_pipeline_picking_layer".into(),
                fragment_entrypoint: "fs_main_picking_layer".into(),
                render_targets: smallvec![Some(PickingLayerProcessor::PICKING_LAYER_FORMAT.into())],
                depth_stencil: PickingLayerProcessor::PICKING_LAYER_DEPTH_STATE,
                multisample: PickingLayerProcessor::PICKING_LAYER_MSAA_STATE,
                ..render_pipeline_desc_color.clone()
            },
        );

        // ---- Outline-mask pipeline ----
        let render_pipeline_outline_mask = render_pipelines.get_or_create(
            ctx,
            &RenderPipelineDesc {
                label: "GaussianSplatRenderer::render_pipeline_outline_mask".into(),
                fragment_entrypoint: "fs_main_outline_mask".into(),
                render_targets: smallvec![Some(OutlineMaskProcessor::MASK_FORMAT.into())],
                depth_stencil: OutlineMaskProcessor::MASK_DEPTH_STATE,
                multisample: OutlineMaskProcessor::mask_default_msaa_state(
                    ctx.device_caps().tier,
                ),
                ..render_pipeline_desc_color
            },
        );

        Self {
            render_pipeline_color,
            render_pipeline_picking_layer,
            render_pipeline_outline_mask,
            bind_group_layout_all_splats,
            bind_group_layout_batch,
        }
    }

    fn draw(
        &self,
        render_pipelines: &GpuRenderPipelinePoolAccessor<'_>,
        phase: DrawPhase,
        pass: &mut wgpu::RenderPass<'_>,
        draw_instructions: &[DrawInstruction<'_, Self::RendererDrawData>],
    ) -> Result<(), DrawError> {
        re_log::info_once!(
        "GaussianSplatRenderer::draw called for phase {:?}, {} instructions",
        phase,
        draw_instructions.len()
        );
        let pipeline_handle = match phase {
            DrawPhase::OutlineMask => self.render_pipeline_outline_mask,
            DrawPhase::Opaque | DrawPhase::Transparent => self.render_pipeline_color,
            DrawPhase::PickingLayer => self.render_pipeline_picking_layer,
            _ => unreachable!("Called on a phase we didn't subscribe to: {phase:?}"),
        };
        let pipeline = match render_pipelines.get(pipeline_handle) {
            Ok(p) => {
                re_log::info_once!("Got pipeline for {:?}", phase);
                p
            }
            Err(e) => {
                re_log::error!("FAILED to get pipeline for {:?}: {:?}", phase, e);
                return Err(e.into());
            }
        };

        pass.set_pipeline(pipeline);

        for DrawInstruction {
            draw_data,
            drawables,
        } in draw_instructions
        {
            let bind_group_all = match phase {
                DrawPhase::OutlineMask => &draw_data.bind_group_all_splats_outline_mask,
                DrawPhase::Opaque | DrawPhase::Transparent | DrawPhase::PickingLayer => &draw_data.bind_group_all_splats,
                _ => unreachable!("Called on a phase we didn't subscribe to: {phase:?}"),
            };
            let Some(bind_group_all) = bind_group_all else {
                debug_assert!(
                    false,
                    "Splat data bind group for phase {phase:?} was not set despite being submitted."
                );
                continue;
            };
            pass.set_bind_group(1, bind_group_all, &[]);

            for drawable in *drawables {
                let batch = &draw_data.batches[drawable.draw_data_payload as usize];
                re_log::info_once!(
                    "Drawing splat batch {:?} vertex_range {:?}",
                    phase,
                    batch.vertex_range
                );
                pass.set_bind_group(2, &batch.bind_group, &[]);
                pass.draw(batch.vertex_range.clone(), 0..1);
            }
        }

        Ok(())
    }
}