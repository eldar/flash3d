import torch
import logging
import time
import torch.nn as nn

from pathlib import Path
from einops import rearrange

from models.encoder.layers import BackprojectDepth
from models.decoder.gauss_util import focal2fov, getProjectionMatrix, K_to_NDC_pp, render_predicted
from misc.util import add_source_frame_id
from misc.depth import estimate_depth_scale, estimate_depth_scale_ransac

def default_param_group(model):
    return [{'params': model.parameters()}]


def to_device(inputs, device):
    for key, ipt in inputs.items():
        if isinstance(ipt, torch.Tensor):
            inputs[key] = ipt.to(device)
    return inputs


class GaussianPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # checking height and width are multiples of 32
        assert cfg.dataset.width % 32 == 0 and cfg.dataset.height % 32 == 0, "'width' and 'height' must be a multiple of 32"

        models = {}
        self.parameters_to_train = []

        # define the model
        if "unidepth" in cfg.model.name:
            from models.encoder.unidepth_encoder import UniDepthExtended
            models["unidepth_extended"] = UniDepthExtended(cfg)
            self.parameters_to_train += models["unidepth_extended"].get_parameter_groups()

        self.models = nn.ModuleDict(models)
        self.set_backproject()

    def set_backproject(self):
        cfg = self.cfg
        backproject_depth = {}
        H = cfg.dataset.height
        W = cfg.dataset.width
        for scale in cfg.model.scales:
            h = H // (2 ** scale)
            w = W // (2 ** scale)
            if cfg.model.shift_rays_half_pixel == "zero":
                shift_rays_half_pixel = 0
            elif cfg.model.shift_rays_half_pixel == "forward":
                shift_rays_half_pixel = 0.5
            elif cfg.model.shift_rays_half_pixel == "backward":
                shift_rays_half_pixel = -0.5
            else:
                raise NotImplementedError
            backproject_depth[str(scale)] = BackprojectDepth(
                cfg.data_loader.batch_size * cfg.model.gaussians_per_pixel, 
                # backprojection can be different if padding was used
                h + 2 * self.cfg.dataset.pad_border_aug, 
                w + 2 * self.cfg.dataset.pad_border_aug,
                shift_rays_half_pixel=shift_rays_half_pixel
            )
        self.backproject_depth = nn.ModuleDict(backproject_depth)

    def target_frame_ids(self, inputs):
        return inputs["target_frame_ids"]
    
    def all_frame_ids(self, inputs):
        return add_source_frame_id(self.target_frame_ids(inputs))

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()
        self._is_train = True

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
        self._is_train = False

    def is_train(self):
        return self._is_train
    
    def forward(self, inputs):
        cfg = self.cfg
        if "unidepth" in cfg.model.name:
            outputs = self.models["unidepth_extended"](inputs)
        
        self.compute_gauss_means(inputs, outputs)

        if cfg.model.gaussian_rendering:
            self.process_gt_poses(inputs, outputs)
            self.render_images(inputs, outputs)

        return outputs
    
    def compute_gauss_means(self, inputs, outputs):
        cfg = self.cfg
        scale = self.cfg.model.scales[0]
        depth = outputs[('depth', scale)]
        B, _, H, W = depth.shape
        inv_K = outputs[("inv_K_src", scale)]
        if self.cfg.model.gaussians_per_pixel > 1:
            inv_K = rearrange(inv_K[:,None,...].
                              repeat(1, self.cfg.model.gaussians_per_pixel, 1, 1),
                              'b n ... -> (b n) ...')
        # back project depth to world splace
        xyz = self.backproject_depth[str(scale)](depth, inv_K)
        if cfg.model.predict_offset:
            offset = outputs["gauss_offset"]
            if cfg.model.scaled_offset:
                offset = offset * depth.detach()
            offset = offset.view(B, 3, -1)
            zeros = torch.zeros(B, 1, H * W, device=depth.device)
            offset = torch.cat([offset, zeros], 1)
            xyz = xyz + offset # [B, 4, W*H]
        inputs[("inv_K_src", scale)] = inv_K
        outputs["gauss_means"] = xyz

    @torch.no_grad()
    def process_gt_poses(self, inputs, outputs):
        cfg = self.cfg
        keyframe = 0
        for f_i in self.target_frame_ids(inputs):
            if ("T_c2w", f_i) not in inputs:
                continue
            T_0 = inputs[("T_c2w", keyframe)]
            T_i = inputs[("T_c2w", f_i)]
            if ("T_w2c", keyframe) in inputs.keys():
                T_0_inv = inputs[("T_w2c", keyframe)]
            else:
                T_0_inv = torch.linalg.inv(T_0.float())
            if ("T_w2c", f_i) in inputs.keys():
                T_i_inv = inputs[("T_w2c", f_i)]
            else:
                T_i_inv = torch.linalg.inv(T_i.float())

            if T_i_inv.dtype == torch.float16 and T_0.dtype == torch.float16:
                outputs[("cam_T_cam", 0, f_i)] = (T_i_inv @ T_0).half()
            else:
                outputs[("cam_T_cam", 0, f_i)] = T_i_inv @ T_0
            if T_0_inv.dtype == torch.float16 and T_i.dtype == torch.float16:
                outputs[("cam_T_cam", f_i, 0)] = (T_0_inv @ T_i).half()
            else:
                outputs[("cam_T_cam", f_i, 0)] = T_0_inv @ T_i


        if cfg.dataset.scale_pose_by_depth:
            B = cfg.data_loader.batch_size
            depth_padded = outputs[("depth", 0)].detach()
            # only use the depth in the unpadded image for scale estimation
            depth = depth_padded[:, :, 
                                 self.cfg.dataset.pad_border_aug:depth_padded.shape[2]-self.cfg.dataset.pad_border_aug,
                                 self.cfg.dataset.pad_border_aug:depth_padded.shape[3]-self.cfg.dataset.pad_border_aug]
            sparse_depth = inputs[("depth_sparse", 0)]
            
            scales = []
            for k in range(B):
                depth_k = depth[[k * self.cfg.model.gaussians_per_pixel], ...]
                sparse_depth_k = sparse_depth[k]
                if ("scale_colmap", 0) in inputs.keys():
                    scale = inputs[("scale_colmap", 0)][k]
                else:
                    if self.is_train():
                        scale = estimate_depth_scale(depth_k, sparse_depth_k)
                    else:
                        scale = estimate_depth_scale_ransac(depth_k, sparse_depth_k)
                scales.append(scale)
            scale = torch.tensor(scales, device=depth.device).unsqueeze(dim=1)
            outputs[("depth_scale", 0)] = scale

            for f_i in self.target_frame_ids(inputs):
                T = outputs[("cam_T_cam", 0, f_i)]
                T[:, :3, 3] = T[:, :3, 3] * scale
                outputs[("cam_T_cam", 0, f_i)] = T
                T = outputs[("cam_T_cam", f_i, 0)]
                T[:, :3, 3] = T[:, :3, 3] * scale
                outputs[("cam_T_cam", f_i, 0)] = T

    def render_images(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        cfg = self.cfg
        B, _, H, W = inputs["color", 0, 0].shape
        for scale in [0]: #cfg.model.scales:
            pos_input_frame = outputs["gauss_means"].float()
            K = inputs[("K_tgt", 0)]
            device = pos_input_frame.device
            dtype = pos_input_frame.dtype

            frame_ids = self.all_frame_ids(inputs)

            for frame_id in frame_ids:
                if frame_id == 0:
                    T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
                else:
                    if ('cam_T_cam', 0, frame_id) not in outputs:
                        continue
                    T = outputs[('cam_T_cam', 0, frame_id)]
                
                if cfg.train.use_gt_poses:
                    pos = pos_input_frame
                else:
                    P = rearrange(T[:, :3, :][:, None, ...].repeat(1, self.cfg.model.gaussians_per_pixel, 1, 1),
                                  'b n ... -> (b n) ...')
                    pos = torch.matmul(P, pos_input_frame)
                
                point_clouds = {
                    "xyz": rearrange(pos[:, :3, :], "(b n) c l -> b (n l) c", n=self.cfg.model.gaussians_per_pixel),
                    "opacity": rearrange(outputs["gauss_opacity"], "(b n) c h w -> b (n h w) c", n=self.cfg.model.gaussians_per_pixel),
                    "scaling": rearrange(outputs["gauss_scaling"], "(b n) c h w -> b (n h w) c", n=self.cfg.model.gaussians_per_pixel),
                    "rotation": rearrange(outputs["gauss_rotation"], "(b n) c h w -> b (n h w) c", n=self.cfg.model.gaussians_per_pixel),
                    "features_dc": rearrange(outputs["gauss_features_dc"], "(b n) c h w -> b (n h w) 1 c", n=self.cfg.model.gaussians_per_pixel)
                }
                if cfg.model.max_sh_degree > 0:
                    point_clouds["features_rest"] = rearrange(outputs["gauss_features_rest"], "(b n) (sh c) h w -> b (n h w) sh c", c=3, n=self.cfg.model.gaussians_per_pixel)

                rgbs = []
                depths = []

                for b in range(B):
                    # get camera projection matrix
                    if cfg.dataset.name in ["kitti", "nyuv2", "waymo"]:
                        K_tgt = inputs[("K_tgt", 0)]
                    else:
                        K_tgt = inputs[("K_tgt", frame_id)]
                    focals_pixels = torch.diag(K_tgt[b])[:2]
                    fovY = focal2fov(focals_pixels[1].item(), H)
                    fovX = focal2fov(focals_pixels[0].item(), W)
                    if cfg.dataset.name in ["co3d", "re10k", "mixed"]:
                        px_NDC, py_NDC = 0, 0
                    else:
                        px_NDC, py_NDC = K_to_NDC_pp(Kx=K_tgt[b][0, 2], Ky=K_tgt[b][1, 2], H=H, W=W)
                    proj_mtrx = getProjectionMatrix(cfg.dataset.znear, cfg.dataset.zfar, fovX, fovY, pX=px_NDC, pY=py_NDC).to(device)
                    world_view_transform = T[b].transpose(0, 1).float()
                    camera_center = (-world_view_transform[3, :3] @ world_view_transform[:3, :3].transpose(0, 1)).float()
                    proj_mtrx = proj_mtrx.transpose(0, 1).float() # [4, 4]
                    full_proj_transform = (world_view_transform@proj_mtrx).float()
                    # use random background for the better opacity learning
                    if cfg.model.randomise_bg_colour and self.is_train():
                        bg_color = torch.rand(3, dtype=dtype, device=device)
                    else:
                        bg_color = torch.tensor(cfg.model.bg_colour, dtype=dtype, device=device)

                    pc = {k: v[b].contiguous().float() for k, v in point_clouds.items()}

                    out = render_predicted(
                        cfg,
                        pc,
                        world_view_transform,
                        full_proj_transform,
                        proj_mtrx,
                        camera_center,
                        (fovX, fovY),
                        (H, W),
                        bg_color,
                        cfg.model.max_sh_degree
                    )
                    rgb = out["render"]
                    rgbs.append(rgb)
                    if "depth" in out:
                        depths.append(out["depth"])
                
                rbgs = torch.stack(rgbs, dim=0)
                outputs[("color_gauss", frame_id, scale)] = rbgs

                if "depth" in out:
                    depths = torch.stack(depths, dim=0)
                    outputs[("depth_gauss", frame_id, scale)] = depths
    
    def checkpoint_dir(self):
        return Path("checkpoints")

    def save_model(self, optimiser, step, ema=None):
        """save model weights to disk"""
        save_folder = self.checkpoint_dir()
        save_folder.mkdir(exist_ok=True, parents=True)

        save_path = save_folder / f"model_{step:07}.pth"
        logging.info(f"saving checkpoint to {str(save_path)}")

        model = ema.ema_model if ema is not None else self
        save_dict = {
            "model": model.state_dict(),
            "version": "1.0",
            "optimiser": optimiser.state_dict(),
            "step": step
        }
        torch.save(save_dict, save_path)

        num_ckpts = self.cfg.run.num_keep_ckpts
        ckpts = sorted(list(save_folder.glob("model_*.pth")), reverse=True)
        if len(ckpts) > num_ckpts:
            for ckpt in ckpts[num_ckpts:]:
                ckpt.unlink()

    def load_model(self, weights_path, optimiser=None, device="cpu", ckpt_ids=0):
        """load model(s) from disk"""
        weights_path = Path(weights_path)

        if weights_path.is_dir():
            ckpts = sorted(list(weights_path.glob("model_*.pth")), reverse=True)
            weights_path = ckpts[ckpt_ids]
        logging.info(f"Loading weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=torch.device(device))
        new_dict = {}
        for k, v in state_dict["model"].items():
            if "backproject_depth" in k:
                new_dict[k] = self.state_dict()[k].clone()
            else:
                new_dict[k] = v.clone()
        self.load_state_dict(new_dict, strict=False)
        
        # loading adam state
        if optimiser is not None:
            optimiser.load_state_dict(state_dict["optimiser"])
            self.step = state_dict["step"]
