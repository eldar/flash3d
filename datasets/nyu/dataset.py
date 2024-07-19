from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from imageio import imread
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as TT
import torchvision.transforms.functional as TTF
from tqdm import tqdm

from datasets.nyu.camera import camera_params, make_K
from datasets.data import pil_loader
from datasets.colmap_utils import read_cameras_binary
from datasets.colmap_misc import\
    read_colmap_pose, \
    load_sparse_pcl_colmap, \
    get_sparse_depth


class NYUv2Dataset(data.Dataset):
    def __init__(self,
                 cfg,
                 split: Optional[str]=None,
                 ):
        super().__init__()
        self.cfg = cfg
        self.data_path = Path(cfg.dataset.data_path)
        self.split = split
        self.color_aug = cfg.dataset.color_aug
        if self.cfg.dataset.pad_border_aug != 0:
            self.pad_border_fn = TT.Pad((self.cfg.dataset.pad_border_aug, self.cfg.dataset.pad_border_aug))
        self.num_scales = 1 # len(cfg.model.scales)
        self.novel_frames = list(cfg.model.gauss_novel_frames)
        self.frame_count = len(self.novel_frames) + 1
        self.max_fov = cfg.dataset.max_fov
        self.interp = Image.LANCZOS

        self.loader = pil_loader
        self.to_tensor = TT.ToTensor()

        self.is_train = False
        
        self.split_name_for_loading = "test"
        self.rgb_file_prefix = "r-"
        self.rgb_file_ext = "ppm"

        split_idxs_path = Path(__file__).resolve().parent / ".." / cfg.dataset.split_path 
        val_items = self._load_split_indices(split_idxs_path)
        val_items = [v for v in val_items if (self.get_colmap_dir(v[0]) / "images.bin").exists()]

        self._seq_keys = sorted(list(set([v[0] for v in val_items])))
        # seq_keys = set([v[0] for v in val_items])
        # valid_seq_keys = []
        # for seq_key in tqdm(seq_keys):
        #     images = read_images_binary(self.get_colmap_dir(seq_key) / "images.bin")
        #     num_recon = len(images.values())
        #     num_images = len(self.get_image_files(seq_key))
        #     if num_recon == num_images:
        #         valid_seq_keys.append(seq_key)
        #     else:
        #         print("COLMAP broken:", seq_key)
        # val_items = [v for v in val_items if v[0] in valid_seq_keys]

        self._seq_key_src_idx_pairs = val_items

        self._skip = 0
        self.length = len(self._seq_key_src_idx_pairs)

    @staticmethod
    def _load_split_indices(index_path):
        def get_key_id(s):
            parts = s.split(" ")
            key = parts[0]
            src_idx = int(parts[1])
            tgt_5_idx = int(parts[2])
            tgt_10_idx = int(parts[3])
            tgt_random_idx = int(parts[4])
                                                      
            return key, [src_idx, tgt_5_idx, tgt_10_idx, tgt_random_idx]

        with open(index_path, "r") as f:
            lines = f.readlines()
        key_id_pairs = list(map(get_key_id, lines))
        return key_id_pairs

    def load_image(self, seq_key, filename, color_aug_fn):
        # load the image
        cfg = self.cfg

        img_file = self.data_path / seq_key / filename

        intr, distortion = camera_params()
        img = imread(img_file)
        H, W = img.shape[:2]

        K = make_K(*intr).copy()

        K_new, roi = cv2.getOptimalNewCameraMatrix(K, distortion, (W,H), 0, (W,H))
        img = cv2.undistort(img, K, distortion, None, K_new)
        x, y, W_undistorted, H_undistorted = roi
        img = img[y:y+H_undistorted, x:x+W_undistorted]

        # correct pp according to the crop
        # focal length remains unchanged in pixels
        K_new[0, 2] -= x
        K_new[1, 2] -= y
        # normalise K by current size
        K_new[0, :] /= W_undistorted
        K_new[1, :] /= H_undistorted

        lft_mrgn = 0
        top_mrgn = 0

        color = self.to_tensor(img)          
        color = TTF.resize(color, (cfg.dataset.height, cfg.dataset.width), 
                            interpolation=TT.InterpolationMode.BICUBIC)
        K_tgt = K_new.copy()
        K_src = K_new.copy()
        K_tgt[0, :] *= cfg.dataset.width 
        K_tgt[1, :] *= cfg.dataset.height

        K_src[0, :] *= cfg.dataset.width 
        K_src[1, :] *= cfg.dataset.height
        K_src[0, 2] += cfg.dataset.pad_border_aug
        K_src[1, 2] += cfg.dataset.pad_border_aug

        color_aug = color_aug_fn(color)
        if self.cfg.dataset.pad_border_aug != 0:
            color_aug = self.pad_border_fn(color_aug)
        return color, color_aug, K_src, K_tgt, (W, H), (lft_mrgn+x, top_mrgn+y)

    def get_colmap_dir(self, seq_key):
        return Path(self.cfg.dataset.colmap_path) / seq_key / "sparse/0"

    def get_image_files(self, seq_key):
        pattern = f"{self.rgb_file_prefix}*.{self.rgb_file_ext}"
        file_paths = self.data_path.joinpath(seq_key).glob(pattern)
        file_names = sorted([p.name for p in file_paths])
        return file_names
    
    def get_full_sequence(self, seq_key, src_idx):
        # get the full sequence of images
        src_and_tgt_frame_idxs = [i for i in range(src_idx, len(self.get_image_files(seq_key)))]
        frame_names = [i-src_idx for i in range(src_idx, len(self.get_image_files(seq_key)))]
        return self.get_inputs_and_targets(seq_key, src_and_tgt_frame_idxs, frame_names)

    def get_inputs_and_targets(self, seq_key, src_and_tgt_frame_idxs, frame_names):
        inputs = {}
        color_aug = (lambda x: x)

        sparse_colmap_dir = self.get_colmap_dir(seq_key)
        sparse_pcl = load_sparse_pcl_colmap(sparse_colmap_dir)
        images = sparse_pcl["images"]

        image_files = self.get_image_files(seq_key)
        image_files = [image_files[i] for i in src_and_tgt_frame_idxs]
    
        # COLMAP is computed on undistorted images with .jpg extension
        image_files_colmap = [f"{Path(f).stem}.jpg" for f in image_files]

        T_w2c_all = []
        for img_name in image_files_colmap:
            colmap_img = [image for image in images.values() if image.name == img_name][0]
            T_w2c = read_colmap_pose(colmap_img)
            T_w2c_all.append(T_w2c)
        
        # load the data
        for frame_name, frame_idx in enumerate(frame_names):
            T_w2c = T_w2c_all[frame_name]
            T_c2w = np.linalg.inv(T_w2c)
            inputs_T_c2w = torch.from_numpy(T_c2w)
            inputs_T_w2c = torch.from_numpy(T_w2c)

            inputs_color, inputs_color_aug, K_src, K_tgt, orig_size, crop_mrgn = \
                self.load_image(seq_key=seq_key, 
                                filename=image_files[frame_idx],
                                color_aug_fn=color_aug
            )

            inv_K_src = np.linalg.pinv(K_src)
            inputs_K_src = torch.from_numpy(K_src)
            inputs_K_tgt = torch.from_numpy(K_tgt)
            inputs_inv_K = torch.from_numpy(inv_K_src)

            global_frame_idx = src_and_tgt_frame_idxs[frame_idx]
            if self.cfg.train.scale_pose_by_depth:
                # get colmap_image_id
                xyd = get_sparse_depth(
                    T_w2c, orig_size, crop_mrgn, sparse_pcl, global_frame_idx
                )
            else:
                xyd = None
            
            inputs[("frame_id", 0)] = \
                f"{seq_key}+{Path(image_files[frame_idx]).stem}"

            if frame_name == 0:
                inputs[("K_src", 0)] = inputs_K_src
                inputs[("inv_K_src", 0)] = inputs_inv_K
                inputs[("K_tgt", 0)] = inputs_K_tgt
            else:
                assert torch.all(inputs[("K_src", 0)] == inputs_K_src)
                assert torch.all(inputs[("inv_K_src", 0)] == inputs_inv_K)
                assert torch.all(inputs[("K_tgt", 0)] == inputs_K_tgt)

            inputs[("color", frame_name, 0)] = inputs_color
            inputs[("color_aug", frame_name, 0)] = inputs_color_aug
            # original world-to-camera matrix in row-major order and transfer to column-major order
            inputs[("T_c2w", frame_name)] = inputs_T_c2w
            inputs[("T_w2c", frame_name)] = inputs_T_w2c
            if xyd is not None:
                inputs[("depth_sparse", frame_name)] = xyd

        return inputs

    def __getitem__(self, index):

        # test data contains pairs of sequence name, [src_idx, tgt_idx1, tgt_idx2, tgt_idx3]
        seq_key, src_and_tgt_frame_idxs = self._seq_key_src_idx_pairs[index]
        # pose_data = self._seq_data[seq_key]

        frame_names = [0, 1, 2, 3]

        return self.get_inputs_and_targets(seq_key, src_and_tgt_frame_idxs, frame_names)
        
    def __len__(self) -> int:
        return self.length
