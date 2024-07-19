import os
import random
import pickle
import gzip
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image
from typing import Optional
from pathlib import Path
from datasets.tardataset import TarDataset

from datasets.data import process_projs, data_to_c2w, pil_loader, get_sparse_depth
from misc.depth import estimate_depth_scale_ransac
from misc.localstorage import copy_to_local_storage, extract_tar, get_local_dir


def load_seq_data(data_path, split):
    file_path = data_path / f"{split}.pickle.gz"
    with gzip.open(file_path, "rb") as f:
        seq_data = pickle.load(f)
    return seq_data


class Re10KDataset(data.Dataset):
    def __init__(self,
                 cfg,
                 split: Optional[str]=None,
                ) -> None:
        super().__init__()

        self.cfg = cfg
        self.data_path = Path(self.cfg.dataset.data_path)
        # if this is a relative path to the code dir, make it absolute
        if not self.data_path.is_absolute(): 
            code_dir = Path(__file__).parents[1]
            relative_path = self.data_path
            self.data_path = code_dir / relative_path
            if not self.data_path.exists():
                raise FileNotFoundError(f"Relative path {relative_path} does not exist")
        elif not self.data_path.exists():
            raise fileNotFoundError(f"Absolute path {self.data_path} does not exist")

        self.depth_path = None
        if self.cfg.dataset.preload_depths:
            assert cfg.dataset.depth_path is not None
            self.depth_path = Path(self.cfg.dataset.depth_path)

        self.split = split
        self.image_size = (self.cfg.dataset.height, self.cfg.dataset.width)
        self.color_aug = self.cfg.dataset.color_aug
        if self.cfg.dataset.pad_border_aug != 0:
            self.pad_border_fn = T.Pad((self.cfg.dataset.pad_border_aug, self.cfg.dataset.pad_border_aug))
        self.num_scales = len(cfg.model.scales)
        self.novel_frames = list(cfg.model.gauss_novel_frames)
        self.frame_count = len(self.novel_frames) + 1
        self.max_fov = cfg.dataset.max_fov
        self.interp = Image.LANCZOS
        self.loader = pil_loader
        self.to_tensor = T.ToTensor()

        self.is_train = self.split == "train"
        
        if self.is_train:
            self.split_name_for_loading = "train"
        else:
            self.split_name_for_loading = "test"

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        # multiple resolution support
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            new_size = (self.image_size[0] // s, self.image_size[1] // s)
            self.resize[i] = T.Resize(new_size, interpolation=self.interp)
        # load dilation file
        self.dilation = cfg.dataset.dilation
        self.max_dilation = cfg.dataset.max_dilation
        if isinstance(self.dilation, int):
            self._left_offset = ((self.frame_count - 1) // 2) * self.dilation
            fixed_dilation = self.dilation
        else: # enters here when cfg.dataset.dilation = random
            self._left_offset = 0
            fixed_dilation = 0

        # load image sequence
        self._seq_data = self._load_seq_data(self.split_name_for_loading)
        self._seq_keys = list(self._seq_data.keys())
        
        if self.is_train:
            self._seq_key_src_idx_pairs = self._full_index(self._seq_keys, 
                self._seq_data, 
                self._left_offset,                    # 0 when sampling dilation randomly
                (self.frame_count-1) * fixed_dilation # 0 when sampling dilation randomly
            )
            if cfg.dataset.subset != -1: # use cfg.dataset.subset source frames, they might come from the same sequence
                self._seq_key_src_idx_pairs = self._seq_key_src_idx_pairs[:cfg.dataset.subset] * (len(self._seq_key_src_idx_pairs) // cfg.dataset.subset)
        else:
            test_split_path = Path(__file__).resolve().parent / ".." / cfg.dataset.test_split_path 
            self._seq_key_src_idx_pairs = self._load_split_indices(test_split_path)

        self.length = len(self._seq_key_src_idx_pairs)
        if cfg.dataset.from_tar and self.is_train:
            fn = self.data_path / "all.train.tar"
            self.images_dataset = TarDataset(archive=fn, extensions=(".jpg", ".pickle"))
            self.pcl_dataset = self.images_dataset
        else:
            fn = self.data_path / f"pcl.{self.split_name_for_loading}.tar"
            if cfg.dataset.copy_to_local:
                pcl_fn = copy_to_local_storage(fn)
                self.pcl_dir = get_local_dir() 
                extract_tar(pcl_fn, self.pcl_dir)
            else:
                self.pcl_dataset = TarDataset(archive=fn, extensions=(".jpg", ".pickle"))

    def __len__(self) -> int:
        return self.length
    
    def _load_seq_data(self, split):
        return load_seq_data(self.data_path, split)
 
    def _full_index(self, seq_keys, seq_data, left_offset, extra_frames):
        skip_bad = self.cfg.dataset.skip_bad_shape
        if skip_bad:
            fn = self.data_path / "valid_seq_ids.train.pickle.gz"
            valid_seq_ids = pickle.load(gzip.open(fn, "rb"))
        key_id_pairs = []
        for seq_key in seq_keys:
            seq_len = len(seq_data[seq_key]["timestamps"])
            frame_ids = [i + left_offset for i in range(seq_len - extra_frames)]
            if skip_bad:
                good_frames = valid_seq_ids[seq_key]
                frame_ids = [f_id for f_id in frame_ids if f_id in good_frames]
            seq_key_id_pairs = [(seq_key, f_id) for f_id in frame_ids]
            key_id_pairs += seq_key_id_pairs
        return key_id_pairs
    
    def _load_sparse_pcl(self, seq_key):
        fn = f"pcl.{self.split_name_for_loading}/{seq_key}.pickle.gz"
        if self.cfg.dataset.from_tar:
            f = self.pcl_dataset.get_file(fn)
            data = gzip.decompress(f.read())
            return pickle.loads(data)
        else:
            fn = self.pcl_dir / fn
            with gzip.open(fn, "rb") as f:
                data = pickle.load(f)
            return data 
        
    def _load_image(self, key, id):
        timestamp = self._seq_data[key]["timestamps"][id]
        path = self.data_path / f"{self.split_name_for_loading}"
        img_file = f"{key}/{timestamp}.jpg"
        if self.cfg.dataset.from_tar and self.is_train:
            img = self.images_dataset.get_image(img_file, pil=True)
            img = img.convert('RGB')
        else:
            img = self.loader(path / img_file)
        return img
    
    def _load_depth(self, key, id):
        timestamp = self._seq_data[key]["timestamps"][id]
        path = self.depth_path / f"{self.split_name_for_loading}"
        depth_file = f"{key}/{timestamp}.png"
        if os.path.exists(path / depth_file):
            depth = Image.open(path / depth_file)
            # Scale the saved image using the metadata
            max_value = float(depth.info["max_value"])
            min_value = float(depth.info["min_value"])
            # Scale from uint16 range
            depth = (np.array(depth).astype(np.float32) / (2 ** 16 - 1)) * (max_value - min_value) + min_value
        else:
            # print("Depth file {} is not exist", path / depth_file)
            depth = None
        return depth
    
    @staticmethod
    def _load_split_indices(index_path):
        "load the testing split from txt"
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

    def get_frame_data(self, seq_key, frame_idx, pose_data, color_aug_fn):
        # load the image
        img = self._load_image(seq_key, frame_idx)
        # load pre-process depth for training
        if self.depth_path is not None:
            depth = self._load_depth(seq_key, frame_idx)
            if depth is not None:
                depth = self.to_tensor(depth)
                depth = F.interpolate(depth[None,...], size=self.image_size, mode="nearest")[0]
        else:
            depth = None

        # load the intrinsics matrix
        K = process_projs(pose_data["intrinsics"][frame_idx])
        # load the extrinsic matrixself.num_scales
        c2w = data_to_c2w(pose_data["poses"][frame_idx])
        img_scale = self.resize[0](img)
        inputs_color = self.to_tensor(img_scale)
        if self.cfg.dataset.pad_border_aug != 0:
            inputs_color_aug = self.to_tensor(color_aug_fn(self.pad_border_fn(img_scale)))
            if depth is not None:
                pad = self.cfg.dataset.pad_border_aug
                depth = F.pad(depth, (pad,pad,pad,pad), mode="replicate")
        else:
            inputs_color_aug = self.to_tensor(color_aug_fn(img_scale))

        K_scale_target = K.copy()
        K_scale_target[0, :] *= self.image_size[1]
        K_scale_target[1, :] *= self.image_size[0]
        # scale K_inv for unprojection according to how much padding was added
        K_scale_source = K.copy()
        # scale focal length by size of original image, scale principal point for the padded image
        K_scale_source[0, 0] *=  self.image_size[1]
        K_scale_source[1, 1] *=  self.image_size[0]
        K_scale_source[0, 2] *= (self.image_size[1] + self.cfg.dataset.pad_border_aug * 2)
        K_scale_source[1, 2] *= (self.image_size[0] + self.cfg.dataset.pad_border_aug * 2)
        inv_K_source = np.linalg.pinv(K_scale_source)

        inputs_K_scale_target = torch.from_numpy(K_scale_target)
        inputs_K_scale_source = torch.from_numpy(K_scale_source)
        inputs_inv_K_source = torch.from_numpy(inv_K_source)

        # original world-to-camera matrix in row-major order and transfer to column-major order
        inputs_T_c2w = torch.from_numpy(c2w)

        return inputs_K_scale_target, inputs_K_scale_source, inputs_inv_K_source, inputs_color, inputs_color_aug, inputs_T_c2w, img.size, depth
    
    def __getitem__(self, index):
        inputs = {}

        # random frame sampling
        if self.is_train:
            # train data contains pairs of sequence name, source frame index
            seq_key, src_idx = self._seq_key_src_idx_pairs[index]
            pose_data = self._seq_data[seq_key]
            seq_len = len(pose_data["timestamps"])

            if self.cfg.dataset.frame_sampling_method == "two_forward_one_back":
                if self.dilation == "random":
                    dilation = torch.randint(1, self.max_dilation, (1,)).item()
                    left_offset = dilation # one frame in the past
                else:
                    # self.dilation and self._left_offsets can be fixed if cfg.dataset.dilation is an int
                    dilation = self.dilation
                    left_offset = self._left_offset
                # frame count is num_novel_frames + 1 for source view
                # sample one frame in backwards time and self.frame_count - 2 into the future
                src_and_tgt_frame_idxs = [src_idx - left_offset + i * dilation for i in range(self.frame_count)]
                # reorder and make sure indices don't go beyond start or end of the sequence
                src_and_tgt_frame_idxs = [src_idx] + [max(min(i, seq_len-1), 0) for i in src_and_tgt_frame_idxs if i != src_idx]
            elif self.cfg.dataset.frame_sampling_method == "random":
                # random indices between -30 and 30 which will mean the offset 
                target_frame_idxs = torch.randperm( 4 * self.max_dilation + 1 )[:self.frame_count] - 2 * self.max_dilation
                # check that 0 is not included and that the indides dont go beyond the end of the sequence
                src_and_tgt_frame_idxs = [src_idx] + [max(min(i + src_idx, seq_len-1), 0) for i in target_frame_idxs.tolist() if i != 0][:self.frame_count - 1]                
            frame_names = [0] + self.novel_frames

        # load src, 5 frames into future, 10 frames into future and random
        # follows MINE split and evaluation protocol
        else:
            # test data contains pairs of sequence name, [src_idx, tgt_idx1, tgt_idx2, tgt_idx3]
            seq_key, src_and_tgt_frame_idxs = self._seq_key_src_idx_pairs[index]
            pose_data = self._seq_data[seq_key]

            frame_names = [0, 1, 2, 3]

        if self.cfg.dataset.scale_pose_by_depth:
            sparse_pcl = self._load_sparse_pcl(seq_key)

        # load the data
        do_color_aug = self.is_train and random.random() > 0.5 and self.color_aug
        if do_color_aug:
            color_aug = T.ColorJitter(
                    self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        for frame_name, frame_idx in zip(frame_names, src_and_tgt_frame_idxs):
            inputs_K_tgt, inputs_K_src, inputs_inv_K_src, inputs_color, inputs_color_aug, \
            inputs_T_c2w, orig_size, inputs_depth = self.get_frame_data(seq_key=seq_key, 
                                                frame_idx=frame_idx, 
                                                pose_data=pose_data,
                                                color_aug_fn=color_aug
            )
            
            if self.cfg.dataset.scale_pose_by_depth:
                # get colmap_image_id
                xyd = get_sparse_depth(pose_data, orig_size, sparse_pcl, frame_idx)
            else:
                xyd = None
            
            input_frame_idx = src_and_tgt_frame_idxs[0]
            timestamp = self._seq_data[seq_key]["timestamps"][input_frame_idx]
            inputs[("frame_id", 0)] = f"{self.split_name_for_loading}+{seq_key}+{timestamp}"

            inputs[("K_tgt", frame_name)] = inputs_K_tgt
            inputs[("K_src", frame_name)] = inputs_K_src
            inputs[("inv_K_src", frame_name)] = inputs_inv_K_src
            inputs[("color", frame_name, 0)] = inputs_color
            inputs[("color_aug", frame_name, 0)] = inputs_color_aug
            # original world-to-camera matrix in row-major order and transfer to column-major order
            inputs[("T_c2w", frame_name)] = inputs_T_c2w
            inputs[("T_w2c", frame_name)] = torch.linalg.inv(inputs_T_c2w)
            if inputs_depth is not None:
                inputs[("unidepth", frame_name, 0)] = inputs_depth

            if xyd is not None and frame_name == 0:
                inputs[("depth_sparse", frame_name)] = xyd

                if inputs_depth is not None and self.cfg.dataset.ransac_on_the_fly:
                    _, H, W = inputs_depth.shape
                    inputs[("scale_colmap", frame_name)] = estimate_depth_scale_ransac(
                        inputs_depth.unsqueeze(0)[:, 
                            self.cfg.dataset.pad_border_aug:H-self.cfg.dataset.pad_border_aug,
                            self.cfg.dataset.pad_border_aug:W-self.cfg.dataset.pad_border_aug],
                        inputs[("depth_sparse", frame_name)]
                    )

        return inputs
