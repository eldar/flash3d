import torch
import numpy as np
from pathlib import Path
from einops import rearrange, einsum
from matplotlib import pyplot as plt
import torch.nn.functional as F
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from torch import Tensor
from scipy.spatial.transform import Rotation as R

from misc.depth import normalize_depth_for_display
from models.encoder.layers import Project3DSimple


def depth_to_img(d):
    d = d.detach().cpu().numpy()
    depth_img = normalize_depth_for_display(d)
    return (np.clip(depth_img, 0, 1) * 255).astype(np.uint8)


def vis_2d_offsets(model, inputs, outputs, out_dir, frame_id):
    input_f_id = 0
    scale = 0
    B, H, W = model.tensor_image_dims()

    xyz = outputs[("gauss_means", input_f_id, scale)]
    K = inputs[("K", scale)]

    p3d = Project3DSimple(1, H, W)
    bp3d = model.backproject_depth[str(scale)]

    pix_coords = p3d(xyz, K)
    pix_coords = rearrange(pix_coords, "1 h w c -> 1 c h w")
    id_coords = rearrange(bp3d.id_coords, "c h w -> 1 c h w")

    s = 8
    pix_coords = F.interpolate(
        pix_coords,
        (H // s, W // s),
        mode='nearest',
    )
    id_coords = F.interpolate(
        id_coords,
        (H // s, W // s),
        mode='nearest',
    )
    v = pix_coords - id_coords
    id_coords = rearrange(id_coords, "1 c h w -> (h w) c")
    v = rearrange(v, "1 c h w -> (h w) c")

    id_coords = id_coords.cpu().numpy()
    v = v.cpu().numpy()

    X = id_coords[:, 0]
    Y = id_coords[:, 1]
    U = v[:, 0]
    V = v[:, 1]
    # print(np.histogram(U)[0], np.histogram(U)[1])

    plt.quiver(X, Y, U, V, color='b', units='xy', scale=1) 
    plt.title('Gauss offset') 

    # x-lim and y-lim 
    plt.xlim(-50, W+50) 
    plt.ylim(-50, H+50) 

    plt.axis('equal')

    # print(mpl.rcParams["savefig.dpi"])

    plt.savefig(out_dir / f"{frame_id}.png", dpi=300.0)
    plt.cla()
    plt.clf()

    # import pdb
    # pdb.set_trace()


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, "gaussian"],
    path: Path,
):
    # Shift the scene so that the median Gaussian is at the origin.
    means = means - means.median(dim=0).values

    # Rescale the scene so that most Gaussians are within range [-1, 1].
    scale_factor = means.abs().quantile(0.95, dim=0).max()
    means = means / scale_factor
    scales = scales / scale_factor

    # Define a rotation that makes +Z be the world up vector.
    rotation = [
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ]
    rotation = torch.tensor(rotation, dtype=torch.float32, device=means.device)

    # The Polycam viewer seems to start at a 45 degree angle. Since we want to be
    # looking directly at the object, we compose a 45 degree rotation onto the above
    # rotation.
    adjustment = torch.tensor(
        R.from_rotvec([0, 0, -45], True).as_matrix(),
        dtype=torch.float32,
        device=means.device,
    )
    rotation = adjustment @ rotation

    # We also want to see the scene in camera space (as the default view). We therefore
    # compose the w2c rotation onto the above rotation.
    # rotation = rotation @ extrinsics[:3, :3].inverse()

    # Apply the rotation to the means (Gaussian positions).
    means = einsum(rotation, means, "i j, ... j -> ... i")

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = rotation.detach().cpu().numpy() @ rotations
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since our axes are swizzled for the spherical harmonics, we only export the DC
    # band.
    harmonics_view_invariant = harmonics

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        harmonics_view_invariant.detach().cpu().contiguous().numpy(),
        opacities.detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def save_ply(outputs, path, gaussians_per_pixel=3):
    means = rearrange(outputs["gauss_means"], "(b v) c n -> b (v n) c", v=gaussians_per_pixel)[0, :, :3]
    scales = rearrange(outputs["gauss_scaling"], "(b v) c h w -> b (v h w) c", v=gaussians_per_pixel)[0]
    rotations = rearrange(outputs["gauss_rotation"], "(b v) c h w -> b (v h w) c", v=gaussians_per_pixel)[0]
    opacities = rearrange(outputs["gauss_opacity"], "(b v) c h w -> b (v h w) c", v=gaussians_per_pixel)[0]
    harmonics = rearrange(outputs["gauss_features_dc"], "(b v) c h w -> b (v h w) c", v=gaussians_per_pixel)[0]

    export_ply(
        means,
        scales,
        rotations,
        harmonics,
        opacities,
        path
    )