// gaussian_splat.wgsl
//
// Renders 3D Gaussian splats as screen-space quads with Gaussian falloff.
// Splat data must be sorted back-to-front by the CPU-side builder for
// correct alpha compositing.

#import <./global_bindings.wgsl>
#import <./types.wgsl>
#import <./utils/flags.wgsl>
#import <./utils/depth_offset.wgsl>

// -----------------------------------------------------------------------
// Bindings
// -----------------------------------------------------------------------

// Per-splat attributes in data textures (pre-sorted back-to-front).
@group(1) @binding(0)
var position_data_texture: texture_2d<f32>;       // xyz = position (object space), w = opacity
@group(1) @binding(1)
var scale_texture: texture_2d<f32>;               // xyz = scale (std-dev per axis), w = unused
@group(1) @binding(2)
var quaternion_texture: texture_2d<f32>;          // xyzw = rotation quaternion (x, y, z, w)
@group(1) @binding(3)
var color_texture: texture_2d<f32>;               // rgba = linear color
@group(1) @binding(4)
var picking_instance_id_texture: texture_2d<u32>; // xy = picking instance id

struct DrawDataUniformBuffer {
    // Reserved for future use (e.g. global opacity multiplier).
    _padding: vec4f,
};

@group(1) @binding(5)
var<uniform> draw_data: DrawDataUniformBuffer;

struct BatchUniformBuffer {
    world_from_obj: mat4x4f,
    depth_offset: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    outline_mask: vec2u,
    picking_layer_object_id: vec2u,
};
@group(2) @binding(0)
var<uniform> batch: BatchUniformBuffer;


// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------

/// Gaussian is clipped beyond this many standard deviations.
const SIGMA_CUTOFF: f32 = 3.0;

/// Small variance added to the projected 2D covariance for anti-aliasing.
const LOW_PASS_VARIANCE: f32 = 0.3;

/// Vertices per splat (two triangles = one quad).
const VERTICES_PER_SPLAT: u32 = 6u;

/// Quad corner offsets in normalised [-1, 1] space.
const QUAD_OFFSETS = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0,  1.0), vec2f(-1.0, 1.0),
);

// -----------------------------------------------------------------------
// Vertex / fragment interface
// -----------------------------------------------------------------------

struct VertexOut {
    @builtin(position)
    position: vec4f,

    /// Pixel-space offset from the splat centre (interpolated across the quad).
    @location(0) @interpolate(linear)
    pixel_offset: vec2f,

    /// xyz = conic (upper triangle of Σ⁻¹: a, b, c  where Σ⁻¹ = [[a,b],[b,c]]),
    /// w   = opacity.
    @location(1) @interpolate(flat)
    conic_and_opacity: vec4f,

    /// Linear RGBA colour (alpha is base colour alpha, NOT opacity).
    @location(2) @interpolate(flat)
    color: vec4f,

    @location(3) @interpolate(flat)
    picking_instance_id: vec2u,
};

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Linear index → 2D texture coordinate.
fn tex_coord(idx: u32, size: vec2u) -> vec2u {
    return vec2u(idx % size.x, idx / size.x);
}

/// Quaternion (x, y, z, w) → column-major rotation matrix.
fn quat_to_mat3(q: vec4f) -> mat3x3f {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let x2 = x + x; let y2 = y + y; let z2 = z + z;
    let xx = x * x2; let xy = x * y2; let xz = x * z2;
    let yy = y * y2; let yz = y * z2; let zz = z * z2;
    let wx = w * x2; let wy = w * y2; let wz = w * z2;

    return mat3x3f(
        vec3f(1.0 - (yy + zz), xy + wz, xz - wy),   // column 0
        vec3f(xy - wz, 1.0 - (xx + zz), yz + wx),    // column 1
        vec3f(xz + wy, yz - wx, 1.0 - (xx + yy)),    // column 2
    );
}

/// Evaluate 2D Gaussian and return combined alpha.
fn gaussian_alpha(offset: vec2f, conic: vec3f, opacity: f32) -> f32 {
    let power = -0.5 * (
        conic.x * offset.x * offset.x +
        conic.z * offset.y * offset.y +
        2.0 * conic.y * offset.x * offset.y
    );
    // exp(-4) ≈ 0.018 — skip negligible contributions early.
    if power < -4.0 { return 0.0; }
    return opacity * exp(power);
}

// -----------------------------------------------------------------------
// Vertex shader
// -----------------------------------------------------------------------

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOut {
    var out: VertexOut;

    let splat_idx  = vertex_idx / VERTICES_PER_SPLAT;
    let corner_idx = vertex_idx % VERTICES_PER_SPLAT;

    // ---- Read per-splat attributes from data textures ----
    let pos_data  = textureLoad(position_data_texture,       tex_coord(splat_idx, textureDimensions(position_data_texture)),       0);
    let scl_data  = textureLoad(scale_texture,               tex_coord(splat_idx, textureDimensions(scale_texture)),               0);
    let quat_data = textureLoad(quaternion_texture,          tex_coord(splat_idx, textureDimensions(quaternion_texture)),          0);
    let col_data  = textureLoad(color_texture,               tex_coord(splat_idx, textureDimensions(color_texture)),               0);
    let pick_data = textureLoad(picking_instance_id_texture, tex_coord(splat_idx, textureDimensions(picking_instance_id_texture)), 0);

    let obj_pos = pos_data.xyz;
    let opacity = pos_data.w;
    let scale   = scl_data.xyz;
    let quat    = quat_data;

    // ---- Transform position to world & view space ----
    let world_pos = (batch.world_from_obj * vec4f(obj_pos, 1.0)).xyz;
    let view_pos  = frame.view_from_world * vec4f(world_pos, 1.0);   // vec3f

    // Cull splats behind or too close to the camera.
    // Rerun uses -Z forward: visible objects have view_pos.z < 0.
    if view_pos.z > -0.01 {
        out.position = vec4f(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    // Positive depth for perspective math.
    let tz = -view_pos.z;

    // Clamp lateral view-space coordinates for numerical stability.
    let lim_x = 1.3 * frame.tan_half_fov.x * tz;
    let lim_y = 1.3 * frame.tan_half_fov.y * tz;
    let tx = clamp(view_pos.x, -lim_x, lim_x);
    let ty = clamp(view_pos.y, -lim_y, lim_y);

    // ---- Focal lengths in pixels ----
    let focal = vec2f(
        frame.framebuffer_resolution.x / (2.0 * frame.tan_half_fov.x),
        frame.framebuffer_resolution.y / (2.0 * frame.tan_half_fov.y),
    );

    // ---- Build Gaussian-local → view-space transform ----
    //
    // M = W_view · W_obj · R · S
    //
    // where R is the splat's rotation and S = diag(scale).
    // This maps from the Gaussian's local scaled frame all the way to
    // camera/view space, so the 2D covariance can be computed as
    //   Σ_2d = (J · M) · (J · M)ᵀ
    // without materialising the intermediate 3×3 world covariance.

    let W_obj = mat3x3f(
        batch.world_from_obj[0].xyz,
        batch.world_from_obj[1].xyz,
        batch.world_from_obj[2].xyz,
    );
    let W_view = mat3x3f(
        frame.view_from_world[0],
        frame.view_from_world[1],
        frame.view_from_world[2],
    );
    let R  = quat_to_mat3(quat);
    let RS = mat3x3f(R[0] * scale.x, R[1] * scale.y, R[2] * scale.z);
    let M  = W_view * W_obj * RS;

    // Jacobian of perspective projection (2×3).
    // Column 2 sign accounts for Rerun's -Z forward convention:
    //   u = focal_x * x / (-z_view)  →  du/dz_view = +focal_x * x / tz²
    let J = mat3x2f(
        vec2f(focal.x / tz, 0.0),
        vec2f(0.0, focal.y / tz),
        vec2f(focal.x * tx / (tz * tz), focal.y * ty / (tz * tz)),
    );

    // ---- Project to 2D covariance: Σ_2d = T · Tᵀ  where T = J · M ----
    let T     = J * M;              // mat3x2f  (2×3 in math)
    let cov2d = T * transpose(T);   // mat2x2f  (2×2 in math)

    // Anti-aliasing low-pass filter (add small isotropic variance).
    let a = cov2d[0].x + LOW_PASS_VARIANCE;
    let b = cov2d[0].y;                        // == cov2d[1].x (symmetric)
    let c = cov2d[1].y + LOW_PASS_VARIANCE;

    let det = a * c - b * b;
    if det <= 0.0 {
        out.position = vec4f(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    // ---- Conic = inverse of 2D covariance (upper triangle) ----
    let inv_det = 1.0 / det;
    let conic   = vec3f(c * inv_det, -b * inv_det, a * inv_det);

    // ---- Bounding quad radius from largest eigenvalue ----
    let mid        = 0.5 * (a + c);
    let lambda_max = mid + sqrt(max(0.0001, mid * mid - det));
    let radius_px  = ceil(SIGMA_CUTOFF * sqrt(lambda_max));

    // ---- Span the screen-space quad ----
    let quad_offset  = QUAD_OFFSETS[corner_idx];
    let pixel_offset = quad_offset * radius_px;

    // Pixel offset → NDC offset.
    // The Jacobian's pixel space is Y-up (positive focal_y),
    // which matches NDC Y-up, so no flip is needed.
    let ndc_offset = pixel_offset * 2.0 / frame.framebuffer_resolution;

    // Project centre to clip space, then offset in NDC.
    let clip_center = frame.projection_from_world * vec4f(world_pos, 1.0);
    out.position = apply_depth_offset(
        vec4f(
            clip_center.xy + ndc_offset * clip_center.w,
            clip_center.z,
            clip_center.w,
        ),
        batch.depth_offset,
    );

    out.pixel_offset        = pixel_offset;
    out.conic_and_opacity   = vec4f(conic, opacity);
    out.color               = col_data;
    out.picking_instance_id = pick_data.xy;

    return out;
}

// -----------------------------------------------------------------------
// Fragment shaders
// -----------------------------------------------------------------------

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    let alpha = gaussian_alpha(
        in.pixel_offset,
        in.conic_and_opacity.xyz,
        in.conic_and_opacity.w,
    );

    if alpha < 1.0 / 255.0 { discard; }

    return vec4f(in.color.rgb * alpha, alpha);

}

@fragment
fn fs_main_picking_layer(in: VertexOut) -> @location(0) vec4u {
    let alpha = gaussian_alpha(
        in.pixel_offset,
        in.conic_and_opacity.xyz,
        in.conic_and_opacity.w,
    );
    if alpha < 0.5 { discard; }

    return vec4u(batch.picking_layer_object_id, in.picking_instance_id);
}

@fragment
fn fs_main_outline_mask(in: VertexOut) -> @location(0) vec2u {
    let alpha = gaussian_alpha(
        in.pixel_offset,
        in.conic_and_opacity.xyz,
        in.conic_and_opacity.w,
    );
    if alpha < 0.5 { discard; }

    return batch.outline_mask;
}
