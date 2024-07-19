from PIL import Image
import numpy as np
import torch


def process_projs(proj):
    # pose in dataset is normalised by resolution
    # need to unnormalise it for metric projection
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = proj[0]
    K[1, 1] = proj[1]
    K[0, 2] = proj[2]
    K[1, 2] = proj[3]
    return K


def pose_to_4x4(w2c):
    if w2c.shape[0] == 3:
        w2c = np.concatenate((w2c.astype(np.float32),
                             np.array([[0, 0, 0, 1]], dtype=np.float32)), axis=0)
    return w2c


def data_to_c2w(w2c):
    w2c = pose_to_4x4(w2c)
    c2w = np.linalg.inv(w2c)
    return c2w


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_sparse_depth(pose_data, orig_size, sparse_pcl, frame_idx):
    # image_id-1 == frame_idx
    xys_all = sparse_pcl["xys"]
    p3D_ids_all = sparse_pcl["p3D_ids"]
    xyz = sparse_pcl["xyz"]

    xys = xys_all[frame_idx]
    p3D_ids = p3D_ids_all[frame_idx]

    W, H = orig_size

    visible_points = p3D_ids != -1
    xys = xys[visible_points, :]
    p3D_ids = p3D_ids[visible_points]

    xyz_image = xyz[p3D_ids, :]
    xyz_image_h = np.hstack((xyz_image, np.ones_like(xyz_image[:, :1])))

    # ===== compute point projections onto image with network data ====
    # index to -1 because image_ids are 1-indexed
    # K = _process_projs(pose_data["intrinsics"][image_id-1], H, W)
    # load the extrinsic matrixself.num_scales
    T_w2c = pose_to_4x4(pose_data["poses"][frame_idx])
    # P = K @ T_w2c
    xyz_pix = np.einsum("ji,ni->nj", T_w2c, xyz_image_h)[:, :3]
    depth = xyz_pix[:, 2:]
    img_dim = np.array([[W, H]])
    xys_scaled = (xys / img_dim - 0.5) * 2
    xyd = np.concatenate([xys_scaled, depth], axis=1)
    return torch.from_numpy(xyd).to(torch.float32)