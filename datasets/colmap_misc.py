import numpy as np
import torch

from datasets.colmap_utils import \
    read_images_binary, \
    read_points3d_binary, \
    read_model, \
    qvec2rotmat


def is_computed(sparse_dir):
    try:
        cameras, images, points3D = read_model(sparse_dir, ".bin")
        is_good = True
    except:
        is_good = False
    return is_good


def read_colmap_pose(image):
    R = qvec2rotmat(image.qvec).astype(np.float32)
    t = image.tvec.astype(np.float32)
    T_w2c = np.vstack([
        np.hstack((R, np.expand_dims(t, axis=1))),
        np.array([0, 0, 0, 1])
    ])
    return T_w2c.astype(np.float32)


def read_camera_params(camera):
    W = camera.width
    H = camera.height
    intr = camera.params
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = intr[0]
    K[1, 1] = intr[1]
    K[0, 2] = intr[2]
    K[1, 2] = intr[3]
    return H, W, K


def load_sparse_pcl_colmap(dir_recon):
    ext = "bin"
    images = read_images_binary(dir_recon / f"images.{ext}")
    points3D = read_points3d_binary(dir_recon / f"points3D.{ext}")

    # convert 3D coordinates to an easier to process format
    xyz_ids = np.array(list(points3D.keys()))
    xyz = np.zeros((np.max(xyz_ids)+1, 3), dtype=np.float32)
    for id in xyz_ids:
        xyz[id, :] = points3D[id].xyz

    image_ids = sorted(list(images.keys()))
    xys = [images[image_id].xys for image_id in image_ids]
    p3D_ids = [images[image_id].point3D_ids for image_id in image_ids]

    return {
        "images": images,
        "xys": xys,
        "p3D_ids": p3D_ids,
        "xyz": xyz
    }


def get_sparse_depth(T_w2c, img_size, crop_margin, sparse_pcl, frame_idx):
    """
    img_size: (W, H) - original size of the image before resizing as used by COLMAP
    """
    # image_id-1 == frame_idx
    xys_all = sparse_pcl["xys"]
    p3D_ids_all = sparse_pcl["p3D_ids"]
    xyz = sparse_pcl["xyz"]

    xys = xys_all[frame_idx]
    p3D_ids = p3D_ids_all[frame_idx]

    visible_points = p3D_ids != -1
    xys = xys[visible_points, :]
    p3D_ids = p3D_ids[visible_points]

    xyz_image = xyz[p3D_ids, :]
    xyz_image_h = np.hstack((xyz_image, np.ones_like(xyz_image[:, :1])))

    # ===== compute point projections onto image with network data ====
    # index to -1 because image_ids are 1-indexed
    # K = _process_projs(pose_data["intrinsics"][image_id-1], H, W)
    # load the extrinsic matrixself.num_scales
    # P = K @ T_w2c
    xyz_pix = np.einsum("ji,ni->nj", T_w2c, xyz_image_h)[:, :3]
    depth = xyz_pix[:, 2:]
    xys_scaled = ((xys - crop_margin) / img_size - 0.5) * 2
    xyd = np.concatenate([xys_scaled, depth], axis=1)
    return torch.from_numpy(xyd).to(torch.float32)