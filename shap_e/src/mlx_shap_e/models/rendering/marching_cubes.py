from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx
import numpy as np
from shap_e.rendering._mc_table import MC_TABLE

_EDGE_CORNERS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


@dataclass(frozen=True)
class McLookupTable:
    cases: mx.array
    masks: mx.array


@lru_cache(maxsize=1)
def _lookup_table() -> McLookupTable:
    cases = np.zeros((256, 5, 3), dtype=np.int32)
    masks = np.zeros((256, 5), dtype=bool)

    edge_to_index = {edge: idx for idx, edge in enumerate(_EDGE_CORNERS)}
    for case_idx, case in enumerate(MC_TABLE):
        for tri_idx, tri in enumerate(case):
            for vert_idx, (c1, c2) in enumerate(zip(tri[::2], tri[1::2])):
                edge = (c1, c2) if c1 < c2 else (c2, c1)
                cases[case_idx, tri_idx, vert_idx] = edge_to_index[edge]
            masks[case_idx, tri_idx] = True

    return McLookupTable(
        cases=mx.array(cases, dtype=mx.int32),
        masks=mx.array(masks, dtype=mx.bool_),
    )


def _create_flat_edge_indices(
    flat_cube_indices: mx.array,
    grid_size: tuple[int, int, int],
) -> mx.array:
    num_xs = (grid_size[0] - 1) * grid_size[1] * grid_size[2]
    y_offset = num_xs
    num_ys = grid_size[0] * (grid_size[1] - 1) * grid_size[2]
    z_offset = num_xs + num_ys
    return mx.stack(
        [
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]
            + flat_cube_indices[:, 1] * grid_size[2]
            + flat_cube_indices[:, 2],
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]
            + (flat_cube_indices[:, 1] + 1) * grid_size[2]
            + flat_cube_indices[:, 2],
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]
            + flat_cube_indices[:, 1] * grid_size[2]
            + flat_cube_indices[:, 2]
            + 1,
            flat_cube_indices[:, 0] * grid_size[1] * grid_size[2]
            + (flat_cube_indices[:, 1] + 1) * grid_size[2]
            + flat_cube_indices[:, 2]
            + 1,
            y_offset
            + flat_cube_indices[:, 0] * (grid_size[1] - 1) * grid_size[2]
            + flat_cube_indices[:, 1] * grid_size[2]
            + flat_cube_indices[:, 2],
            y_offset
            + (flat_cube_indices[:, 0] + 1) * (grid_size[1] - 1) * grid_size[2]
            + flat_cube_indices[:, 1] * grid_size[2]
            + flat_cube_indices[:, 2],
            y_offset
            + flat_cube_indices[:, 0] * (grid_size[1] - 1) * grid_size[2]
            + flat_cube_indices[:, 1] * grid_size[2]
            + flat_cube_indices[:, 2]
            + 1,
            y_offset
            + (flat_cube_indices[:, 0] + 1) * (grid_size[1] - 1) * grid_size[2]
            + flat_cube_indices[:, 1] * grid_size[2]
            + flat_cube_indices[:, 2]
            + 1,
            z_offset
            + flat_cube_indices[:, 0] * grid_size[1] * (grid_size[2] - 1)
            + flat_cube_indices[:, 1] * (grid_size[2] - 1)
            + flat_cube_indices[:, 2],
            z_offset
            + (flat_cube_indices[:, 0] + 1) * grid_size[1] * (grid_size[2] - 1)
            + flat_cube_indices[:, 1] * (grid_size[2] - 1)
            + flat_cube_indices[:, 2],
            z_offset
            + flat_cube_indices[:, 0] * grid_size[1] * (grid_size[2] - 1)
            + (flat_cube_indices[:, 1] + 1) * (grid_size[2] - 1)
            + flat_cube_indices[:, 2],
            z_offset
            + (flat_cube_indices[:, 0] + 1) * grid_size[1] * (grid_size[2] - 1)
            + (flat_cube_indices[:, 1] + 1) * (grid_size[2] - 1)
            + flat_cube_indices[:, 2],
        ],
        axis=-1,
    )


def _compact_rows_by_mask(values: mx.array, mask: mx.array) -> mx.array:
    mask_i = mask.astype(mx.int32)
    num_selected = int(mx.sum(mask_i))
    if num_selected == 0:
        return values[:0]

    true_rank = mx.cumsum(mask_i, axis=0) - 1
    false_rank = mx.cumsum((1 - mask_i).astype(mx.int32), axis=0) - 1
    destinations = mx.where(mask, true_rank, false_rank + num_selected).astype(mx.int32)
    dest_indices = mx.broadcast_to(destinations[:, None], values.shape)
    compacted = mx.put_along_axis(mx.zeros(values.shape, dtype=values.dtype), dest_indices, values, axis=0)
    return compacted[:num_selected]


def _compact_vector_by_mask(values: mx.array, mask: mx.array) -> mx.array:
    mask_i = mask.astype(mx.int32)
    num_selected = int(mx.sum(mask_i))
    if num_selected == 0:
        return values[:0]

    true_rank = mx.cumsum(mask_i, axis=0) - 1
    false_rank = mx.cumsum((1 - mask_i).astype(mx.int32), axis=0) - 1
    destinations = mx.where(mask, true_rank, false_rank + num_selected).astype(mx.int32)
    compacted = mx.put_along_axis(mx.zeros(values.shape, dtype=values.dtype), destinations, values, axis=0)
    return compacted[:num_selected]


def marching_cubes(
    field,
    min_point,
    size,
) -> tuple[mx.array, mx.array]:
    """
    Deterministic MLX marching-cubes implementation that mirrors the
    Shap-E torch helper's edge indexing and interpolation math.
    """
    field = field.astype(mx.float32) if isinstance(field, mx.array) else mx.array(field, dtype=mx.float32)
    if field.ndim != 3:
        raise ValueError(f"field must be 3D, got shape {field.shape}")

    grid_size = tuple(int(v) for v in field.shape)
    if min(grid_size) < 2:
        return mx.zeros((0, 3), dtype=mx.float32), mx.zeros((0, 3), dtype=mx.int32)

    lut = _lookup_table()

    bitmasks = (field > 0).astype(mx.uint8)
    bitmasks = bitmasks[:-1, :, :] | (bitmasks[1:, :, :] << 1)
    bitmasks = bitmasks[:, :-1, :] | (bitmasks[:, 1:, :] << 2)
    bitmasks = bitmasks[:, :, :-1] | (bitmasks[:, :, 1:] << 4)

    xs = mx.broadcast_to(mx.arange(grid_size[0], dtype=mx.float32)[:, None, None], grid_size)
    ys = mx.broadcast_to(mx.arange(grid_size[1], dtype=mx.float32)[None, :, None], grid_size)
    zs = mx.broadcast_to(mx.arange(grid_size[2], dtype=mx.float32)[None, None, :], grid_size)
    corner_coords = mx.stack([xs, ys, zs], axis=-1)

    edge_midpoints = mx.concatenate(
        [
            ((corner_coords[:-1] + corner_coords[1:]) / 2.0).reshape(-1, 3),
            ((corner_coords[:, :-1] + corner_coords[:, 1:]) / 2.0).reshape(-1, 3),
            ((corner_coords[:, :, :-1] + corner_coords[:, :, 1:]) / 2.0).reshape(-1, 3),
        ],
        axis=0,
    )

    cube_indices = corner_coords[:-1, :-1, :-1].astype(mx.int32).reshape(-1, 3)
    edge_indices = _create_flat_edge_indices(cube_indices, grid_size)

    flat_bitmasks = bitmasks.reshape(-1).astype(mx.int32)
    local_tris = lut.cases[flat_bitmasks]
    local_masks = lut.masks[flat_bitmasks]
    if not bool(mx.any(local_masks)):
        return mx.zeros((0, 3), dtype=mx.float32), mx.zeros((0, 3), dtype=mx.int32)

    global_tris = mx.take_along_axis(
        edge_indices,
        local_tris.reshape(local_tris.shape[0], -1),
        axis=1,
    ).reshape(local_tris.shape)
    selected_tris = _compact_rows_by_mask(global_tris.reshape(-1, 3), local_masks.reshape(-1))
    if int(selected_tris.size) == 0:
        return mx.zeros((0, 3), dtype=mx.float32), mx.zeros((0, 3), dtype=mx.int32)

    flat_selected = selected_tris.reshape(-1)
    order = mx.argsort(flat_selected)
    sorted_used = flat_selected[order]
    is_first = mx.ones((1,), dtype=mx.bool_)
    is_new = mx.concatenate([is_first, sorted_used[1:] != sorted_used[:-1]], axis=0)
    used_vertex_indices = _compact_vector_by_mask(sorted_used, is_new)
    used_edge_midpoints = edge_midpoints[used_vertex_indices]
    compact_ids_sorted = mx.cumsum(is_new.astype(mx.int32), axis=0) - 1
    flat_faces = mx.put_along_axis(
        mx.zeros(flat_selected.shape, dtype=mx.int32),
        order,
        compact_ids_sorted.astype(mx.int32),
        axis=0,
    )
    faces = flat_faces.reshape(selected_tris.shape)

    v1 = mx.floor(used_edge_midpoints).astype(mx.int32)
    v2 = mx.ceil(used_edge_midpoints).astype(mx.int32)
    s1 = field[v1[:, 0], v1[:, 1], v1[:, 2]]
    s2 = field[v2[:, 0], v2[:, 1], v2[:, 2]]

    grid_size_arr = mx.array(grid_size, dtype=mx.float32)
    min_point = min_point.astype(mx.float32) if isinstance(min_point, mx.array) else mx.array(min_point, dtype=mx.float32)
    size = size.astype(mx.float32) if isinstance(size, mx.array) else mx.array(size, dtype=mx.float32)
    p1 = (v1.astype(mx.float32) / (grid_size_arr - 1.0)) * size + min_point
    p2 = (v2.astype(mx.float32) / (grid_size_arr - 1.0)) * size + min_point

    t = (s1 / (s1 - s2))[:, None]
    verts = t * p2 + (1.0 - t) * p1
    return verts.astype(mx.float32), faces.astype(mx.int32)
