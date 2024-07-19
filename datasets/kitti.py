import os
import random
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T

from PIL import Image
from typing import Optional
from pathlib import Path
from datasets.data import  pil_loader


# This could also be retrieved from
BASE_SIZES = {
    "2011_09_26": (375, 1242),
    "2011_09_28": (370, 1224),
    "2011_09_29": (374, 1238),
    "2011_09_30": (370, 1226),
    "2011_10_03": (376, 1241),
}

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

class KITTIDataset(data.Dataset):
    def __init__(self,
                 cfg,
                 split: Optional[str]=None,
                 ):
        super().__init__()

        self.cfg = cfg
        self.data_path = Path(self.cfg.dataset.data_path)
        self.split = split
        fpath = os.path.join(cfg.dataset.split_path, cfg.dataset.split, f"{split}_files.txt")
        self.filenames = readlines(fpath)

        self.image_size = (self.cfg.dataset.height, self.cfg.dataset.width)
        if self.cfg.dataset.pad_border_aug != 0:
            self.pad_border_fn = T.Pad((self.cfg.dataset.pad_border_aug, 
                                        self.cfg.dataset.pad_border_aug))
        self.num_scales = len(cfg.model.scales)
        self.interp = Image.LANCZOS
        self.loader = pil_loader
        self.to_tensor = T.ToTensor()
        
        if cfg.model.gaussian_rendering:
            frame_idxs = [0] + cfg.model.gauss_novel_frames
            if cfg.dataset.stereo:
                if split == "train":
                    stereo_frames = []
                    for frame_id in frame_idxs:
                        stereo_frames += [f"s{frame_id}"]
                    frame_idxs += stereo_frames
                else:
                    frame_idxs = [0, "s0"]
        else:
            # SfMLearner frames, eg. [0, -1, 1]
            frame_idxs = cfg.model.frame_ids.copy()
        self.frame_idxs = frame_idxs

        self.is_train = split == "train"
        self.img_ext = '.png' if cfg.dataset.png else '.jpg'

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

        self.resize_depth = T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self._calibs = self._load_calibs(self.data_path, self.image_size, cfg.dataset.keep_aspect_ratio)
        self._sequences = self._get_sequences(self.data_path)
        self.pose_path = cfg.dataset.pose_path
        self.depth_path = cfg.dataset.depth_path
        self.gt_depths = True if self.depth_path is not None else False
        self.gt_poses = True if self.pose_path is not None else False
        if self.pose_path is not None:
            self._poses = self._load_poses(self.pose_path, self._sequences)

    def __len__(self):
        return len(self.filenames)

    def get_color(self, folder, frame_index, side, do_flip):
        image_path = self.get_image_path(folder, frame_index, side)
        color = self.loader(image_path)

        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color

    def get_depth_anything(self, folder, frame_index, side, do_flip):
        f_str = f"{frame_index:010d}.npy"
        depth_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)

        depth_gt = np.squeeze(np.load(depth_path))

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                if self.cfg.dataset.pad_border_aug != 0:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(self.pad_border_fn(color_aug(f)))
                else:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    @staticmethod
    def _get_sequences(data_path):
        all_sequences = []

        data_path = Path(data_path)
        for day in data_path.iterdir():
            if not day.is_dir():
                continue
            day_sequences = [seq for seq in day.iterdir() if seq.is_dir()]
            # lengths = [len(list((seq / "image_02" / "data").iterdir())) for seq in day_sequences]
            # day_sequences = [(day.name, seq.name, length) for seq, length in zip(day_sequences, lengths)]
            day_sequences = [(day.name, seq.name) for seq in day_sequences]
            all_sequences.extend(day_sequences)

        return all_sequences

    @staticmethod
    def _load_poses(pose_path, sequences):
        poses = {}

        # for day, seq, _ in sequences:
        for day, seq in sequences:
            pose_file = Path(pose_path) / day / f"{seq}.txt"

            poses_seq = []
            try:
                with open(pose_file, 'r') as f:
                    lines = f.readlines()

                    for line in lines:
                        T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                        T_w_cam0 = T_w_cam0.reshape(3, 4)
                        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                        poses_seq.append(T_w_cam0)

            except FileNotFoundError:
                pass
                # print(f'Ground truth poses are not available for sequence {seq}.')

            poses_seq = np.array(poses_seq, dtype=np.float32)

            poses[(day, seq)] = poses_seq
        return poses        

    @staticmethod
    def _load_calibs(data_path, target_image_size, keep_aspect_ratio):
        calibs = {}

        for day in BASE_SIZES.keys():
            day_folder = Path(data_path) / day
            cam_calib_file = day_folder / "calib_cam_to_cam.txt"
            velo_calib_file = day_folder / "calib_velo_to_cam.txt"

            cam_calib_file_data = {}
            with open(cam_calib_file, 'r') as f:
                for line in f.readlines():
                    key, value = line.split(':', 1)
                    try:
                        cam_calib_file_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                    except ValueError:
                        pass
            velo_calib_file_data = {}
            with open(velo_calib_file, 'r') as f:
                for line in f.readlines():
                    key, value = line.split(':', 1)
                    try:
                        velo_calib_file_data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)
                    except ValueError:
                        pass

            im_size = BASE_SIZES[day]

            # Create 3x4 projection matrices
            P_rect_l = np.reshape(cam_calib_file_data['P_rect_02'], (3, 4))
            P_rect_r = np.reshape(cam_calib_file_data['P_rect_03'], (3, 4))

            R_rect = np.eye(4, dtype=np.float32)
            R_rect[:3, :3] = cam_calib_file_data['R_rect_00'].reshape(3, 3)

            T_v2c = np.hstack((velo_calib_file_data['R'].reshape(3, 3), velo_calib_file_data['T'][..., np.newaxis]))
            T_v2c = np.vstack((T_v2c, np.array([0, 0, 0, 1.0], dtype=np.float32)))

            P_v2cl = P_rect_l @ R_rect @ T_v2c
            P_v2cr = P_rect_r @ R_rect @ T_v2c

            # Compute the rectified extrinsics from cam0 to camN
            T_l = np.eye(4, dtype=np.float32)
            T_l[0, 3] = P_rect_l[0, 3] / P_rect_l[0, 0]
            T_r = np.eye(4, dtype=np.float32)
            T_r[0, 3] = P_rect_r[0, 3] / P_rect_r[0, 0]

            K = P_rect_l[:3, :3]

            if keep_aspect_ratio:
                r_orig = im_size[0] / im_size[1]
                r_target = target_image_size[0] / target_image_size[1]

                if r_orig >= r_target:
                    new_height = r_target * im_size[1]
                    crop_height = im_size[0] - ((im_size[0] - new_height) // 2) * 2
                    box = ((im_size[0] - new_height) // 2, 0, crop_height, int(im_size[1]))

                    c_x = K[0, 2] / im_size[1]
                    c_y = (K[1, 2] - (im_size[0] - new_height) / 2) / new_height

                    rescale = im_size[1] / target_image_size[1]

                else:
                    new_width = im_size[0] / r_target
                    crop_width = im_size[1] - ((im_size[1] - new_width) // 2) * 2
                    box = (0, (im_size[1] - new_width) // 2, im_size[0], crop_width)

                    c_x = (K[0, 2] - (im_size[1] - new_width) / 2) / new_width
                    c_y = K[1, 2] / im_size[0]

                    rescale = im_size[0] / target_image_size[0]

                f_x = (K[0, 0] / target_image_size[1]) / rescale
                f_y = (K[1, 1] / target_image_size[0]) / rescale

                box = tuple([int(x) for x in box])

            else:
                f_x = K[0, 0] / im_size[1]
                f_y = K[1, 1] / im_size[0]

                c_x = K[0, 2] / im_size[1]
                c_y = K[1, 2] / im_size[0]

                box = None

            # Replace old K with new K
            K[0, 0] = f_x * 2.
            K[1, 1] = f_y * 2.
            K[0, 2] = c_x * 2 - 1
            K[1, 2] = c_y * 2 - 1

            K_raw = np.eye(4, dtype=np.float32)
            K_raw[0, 0] = f_x
            K_raw[1, 1] = f_y
            K_raw[0, 2] = c_x
            K_raw[1, 2] = c_y

            # Invert to get camera to center transformation, not center to camera
            T_r = np.linalg.inv(T_r)
            T_l = np.linalg.inv(T_l)

            calibs[day] = {
                "K": K,
                "K_raw": K_raw,
                "T_l": T_l,
                "T_r": T_r,
                "P_v2cl": P_v2cl,
                "P_v2cr": P_v2cr,
                "crop": box
            }

        return calibs
    
    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K_tgt", scale)                        for camera intrinsics when projecting,
            ("inv_K", scale)                        for camera intrinsics when unprojecting,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        cfg = self.cfg

        inputs = {}

        do_color_aug = cfg.dataset.color_aug and self.is_train and random.random() > 0.5
        do_flip = cfg.dataset.flip_left_right and self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]
        day, sequence = folder.split("/")
        calibs = self._calibs[day]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        stereo_flip = {"r": "l", "l": "r"}

        if len(line) == 3:
            side = line[2]
            flip_stereo = self.is_train and random.random() > 0.5
            if flip_stereo:
                side = stereo_flip[side]
        else:
            side = None

        frame_idxs = list(self.frame_idxs).copy()

        for f_id in frame_idxs:
            if type(f_id) == str and f_id[0] == "s": # stereo frame
                the_side = stereo_flip[side]
                i = int(f_id[1:])
            else:
                the_side = side
                i = f_id
            inputs[("color", f_id, -1)] = self.get_color(folder, frame_index + i, the_side, do_flip)
        
        inputs[("frame_id", 0)] = \
            f"{os.path.split(folder)[1]}+{side}+{frame_index:06d}"

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            if self.cfg.dataset.precise_intrinsics:
                K = calibs["K_raw"]
            else:
                K = self.K

            K_tgt = K.copy()
            K_src = K.copy()

            assert not do_flip
            if self.cfg.dataset.precise_intrinsics and do_flip:
                K[0, 2] = 1.0 - K[0, 2]

            K_tgt[0, :] *= self.image_size[1] // (2 ** scale)
            K_tgt[1, :] *= self.image_size[0] // (2 ** scale)

            K_src[0, :] *= self.image_size[1] // (2 ** scale)
            K_src[1, :] *= self.image_size[0] // (2 ** scale)
            # principal points change if we add padding
            K_src[0, 2] += self.cfg.dataset.pad_border_aug // (2 ** scale)
            K_src[1, 2] += self.cfg.dataset.pad_border_aug // (2 ** scale)

            inv_K_src = np.linalg.pinv(K_src)

            inputs[("K_tgt", scale)] = torch.from_numpy(K_tgt)[..., :3, :3]
            inputs[("K_src", scale)] = torch.from_numpy(K_src)[..., :3, :3]
            inputs[("inv_K_src", scale)] = torch.from_numpy(inv_K_src)[..., :3, :3]

        if do_color_aug:
            raise NotImplementedError
            color_aug = random_color_jitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.gt_depths:
            depth_gt = self.get_depth_anything(folder, frame_index, side, do_flip)
            depth_gt = np.expand_dims(depth_gt, 0)
            depth_gt = torch.from_numpy(depth_gt.astype(np.float32))
            depth_gt = self.resize_depth(depth_gt)
            inputs[("depth_gt", 0, 0)] = depth_gt

        if self.gt_poses:
            # Load "GT" poses
            for f_id in frame_idxs:
                if type(f_id) == str and f_id[0] == "s": # stereo frame
                    the_side = {"r": "l", "l": "r"}[side]
                    i = int(f_id[1:])
                else:
                    the_side = side
                    i = f_id
                id = frame_index + i
                T_side = calibs[f"T_{the_side}"]
                pose = self._poses[(day, sequence)][id, :, :] @ T_side
                inputs[("T_c2w", f_id)] = pose

        return inputs

