#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def getProjectionMatrix(znear, zfar, fovX, fovY, pX=0.0, pY=0.0):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # camera coordinates are x right, y down, z away from camera
    # assume that at this point principal points are in 
    # (x right, y down) convention in NDC coordinates
    # +ve principal point means it's down / to the right of center
    # point at (0, 0, 1) should get imaged to (px, py) in 
    top    = tanHalfFovY * znear * ( 1 + pY)
    bottom = tanHalfFovY * znear * (-1 + pY)
    right  = tanHalfFovX * znear * ( 1 + pX)
    left   = tanHalfFovX * znear * (-1 + pX)

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def K_to_NDC_pp(Kx, Ky, H, W):
    # Kx is in pixels from the left border
    # Ky is in pixels from the top border
    # H, W are in pixels
    # transform to NDC in which image borders are at (-1, 1)
    # positive pp is to the right, down
    px = 2 * Kx / W - 1
    py = 2 * Ky / H - 1
    return px, py

def render_predicted(cfg,
                     pc : dict, 
                     world_view_transform,
                     full_proj_transform,
                     proj_mtrx,
                     camera_center,
                     fov,
                     img_size, 
                     bg_color : torch.Tensor, 
                     max_sh_degree,
                     scaling_modifier = 1.0, 
                     override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc["xyz"], dtype=pc["xyz"].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    fovX, fovY = fov
    tanfovx = math.tan(fovX * 0.5)
    tanfovy = math.tan(fovY * 0.5)

    height, width = img_size

    kvargs = {
        "image_height": height,
        "image_width": width,
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
        "bg": bg_color,
        "scale_modifier": scaling_modifier,
        "viewmatrix": world_view_transform,
        "projmatrix": full_proj_transform,
        "sh_degree": max_sh_degree,
        "campos": camera_center,
        "prefiltered": False,
        "debug": False
    }
    if cfg.model.renderer_w_pose:
        kvargs |= {"projmatrix_raw": proj_mtrx}

    # Set up rasterization configuration
    raster_settings = GaussianRasterizationSettings(**kvargs)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc["xyz"] # .contiguous()
    means2D = screenspace_points # .contiguous()
    opacity = pc["opacity"] # .contiguous()

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc["scaling"] # .contiguous()
    rotations = pc["rotation"] # .contiguous()

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if "features_rest" in pc.keys():
            shs = torch.cat([pc["features_dc"], pc["features_rest"]], dim=1).contiguous()
        else:
            shs = pc["features_dc"] # .contiguous()
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    outputs = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )
    rendered_image, radii = outputs[:2]
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    output = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii
    }
    if len(outputs) >= 4:
        rendered_depth, rendered_alpha = outputs[2:4]
        output["depth"] = rendered_depth
        output["opacity"] = opacity
    if len(outputs) >= 5:
        n_touched = outputs[4]

    return output
