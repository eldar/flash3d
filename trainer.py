import time
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

from models.model import GaussianPredictor
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from misc.depth import normalize_depth_for_display
from misc.util import sec_to_hm_str

from models.encoder.layers import SSIM
from evaluate import evaluate, get_model_instance


class Trainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.step = 0
        self.model = GaussianPredictor(cfg)
        if cfg.loss.ssim.weight > 0:
            self.ssim = SSIM()
        if cfg.loss.lpips.weight > 0:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def forward(self, inputs):
        outputs = self.model.forward(inputs)
        losses = self.compute_losses(inputs, outputs)
        return losses, outputs
    
    def compute_reconstruction_loss(self, pred, target, losses):
        """Computes reprojection loss between a batch of predicted and target images
        """
        cfg = self.cfg
        rec_loss = 0.0
        # pixel level loss
        if cfg.loss.mse.weight > 0:
            if cfg.loss.mse.type == "l1":
                mse_loss = (pred-target).abs().mean()
            elif cfg.loss.mse.type == "l2":
                mse_loss = ((pred-target)**2).mean()
            losses["loss/mse"] = mse_loss
            rec_loss += cfg.loss.mse.weight * mse_loss
        # patch level loss
        if cfg.loss.ssim.weight > 0:
            ssim_loss = self.ssim(pred, target).mean()
            losses["loss/ssim"] = ssim_loss
            rec_loss += cfg.loss.ssim.weight * ssim_loss
        # feature level loss
        if cfg.loss.lpips.weight > 0:
            if self.step > cfg.loss.lpips.apply_after_step:
                lpips_loss = self.lpips.to(pred.device)((pred * 2 - 1).clamp(-1,1), 
                                   (target * 2 - 1).clamp(-1,1))
                losses["loss/lpips"] = lpips_loss
                rec_loss += cfg.loss.lpips.weight * lpips_loss
        
        return rec_loss
    
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        cfg = self.cfg
        losses = {}
        total_loss = 0.0

        if cfg.model.gaussian_rendering:
            # regularize too big or too small gaussians
            if (big_g_lmbd := cfg.loss.gauss_scale.weight) > 0:
                scaling = outputs["gauss_scaling"]
                big_gaussians = torch.where(scaling > cfg.loss.gauss_scale.thresh)
                if len(big_gaussians[0]) > 0:
                    big_gauss_reg_loss = torch.mean(scaling[big_gaussians])
                else:
                    big_gauss_reg_loss = 0
                losses["loss/big_gauss_reg_loss"] = big_gauss_reg_loss
                total_loss += big_g_lmbd * big_gauss_reg_loss
            
            # regularize too big offset
            if cfg.model.predict_offset and (offs_lmbd := cfg.loss.gauss_offset.weight) > 0:
                offset = outputs["gauss_offset"]
                big_offset = torch.where(offset**2 > cfg.loss.gauss_offset.thresh**2)
                if len(big_offset[0]) > 0:
                    big_offset_reg_loss = torch.mean(offset[big_offset]**2)
                else:
                    big_offset_reg_loss = 0.0
                losses["loss/gauss_offset_reg"] = big_offset_reg_loss
                total_loss += offs_lmbd * big_offset_reg_loss

            # reconstruction loss
            frame_ids = self.model.all_frame_ids(inputs)
            rec_loss = 0
            for frame_id in frame_ids:
                # compute gaussian reconstruction loss
                target = inputs[("color_aug", frame_id, 0)]
                target = target[:,:,cfg.dataset.pad_border_aug:target.shape[2]-cfg.dataset.pad_border_aug,
                                cfg.dataset.pad_border_aug:target.shape[3]-cfg.dataset.pad_border_aug,]
                pred = outputs[("color_gauss", frame_id, 0)]
                rec_loss += self.compute_reconstruction_loss(pred, target, losses)
            rec_loss /= len(frame_ids)
            losses["loss/rec"] = rec_loss
            total_loss += rec_loss

        losses["loss/total"] = total_loss
        return losses
    
    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.cfg.optimiser.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
    
    def log_scalars(self, mode, outputs, losses, lr):
        """log the scalars"""
        cfg = self.cfg
        logger = self.logger
        if logger is None:
            return

        logger.log({f"{mode}/learning_rate": lr}, self.step)
        logger.log({f"{mode}/{l}": v for l, v in losses.items()}, self.step)
        if cfg.model.gaussian_rendering:
            logger.log({f"{mode}/gauss/scale/mean": torch.mean(outputs["gauss_scaling"])}, self.step)

            if self.cfg.model.predict_offset:
                offset_mag = torch.linalg.vector_norm(outputs["gauss_offset"], dim=1)
                mean_offset = offset_mag.mean()
                logger.log({f"{mode}/gauss/offset/mean": mean_offset}, self.step)
        if cfg.dataset.scale_pose_by_depth:
            depth_scale = outputs[("depth_scale", 0)]
            logger.log({f"{mode}/depth_scale": depth_scale.mean().item()}, self.step)

    def log(self, mode, inputs, outputs):
        """Write images to Neptune
        """
        cfg = self.cfg
        frame_ids = self.model.all_frame_ids(inputs)
        scales = cfg.model.scales
        logger = self.logger
        if logger is None:
            return

        for j in range(min(4, cfg.data_loader.batch_size)): # write a maxmimum of 4 images
            for s in scales:
                assert cfg.model.gaussian_rendering
                for frame_id in frame_ids:
                    logger.log_image(
                        f"{mode}/color_gauss/{j}/gt_aug/{frame_id}",
                        inputs[("color_aug", frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy(),
                        self.step
                    )
                
                for frame_id in frame_ids:
                    logger.log_image(
                        f"{mode}/color_gauss/{j}/gt/{frame_id}",
                        inputs[("color", frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy(),
                        self.step
                    )

                for frame_id in frame_ids:
                    logger.log_image(
                        f"{mode}/color_gauss/{j}/pred/{frame_id}",
                        outputs[("color_gauss", frame_id, 0)][j].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy(),
                        self.step
                    )

                for i in range(self.cfg.model.gaussians_per_pixel):
                    logger.log_image(
                        f"{mode}/gauss_opacity_gaussian_{i}/{j}",
                        outputs["gauss_opacity"][j * self.cfg.model.gaussians_per_pixel + i].data.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy(),
                        self.step
                    )

                depth = rearrange(outputs[("depth", 0)], "(b n) ... -> b n ...", n=self.cfg.model.gaussians_per_pixel)
                depth_sliced = depth[j][0].detach().cpu().numpy()
                depth_img, normalizer = normalize_depth_for_display(depth_sliced, return_normalizer=True)
                depth_img = np.clip(depth_img, 0, 1)

                logger.log_image(f"{mode}/depth_{s}/{j}", depth_img, self.step)

                for layer in range(1, self.cfg.model.gaussians_per_pixel):
                    depth_sliced = depth[j][layer].detach().cpu().numpy()
                    depth_img =  normalize_depth_for_display(depth_sliced, normalizer=normalizer)
                    depth_img = np.clip(depth_img, 0, 1)
                    logger.log_image(
                        f"{mode}/depth_{layer}_gaussian_{s}/{j}",
                        depth_img,
                        self.step
                    )

    def validate(self, model, evaluator, val_loader, device):
        """
        model may not be the same as trainer, in case of wrapping it in EMA
        sets model to eval mode by evaluate()
        """
        score_dict_by_name = evaluate(model, self.cfg, evaluator, val_loader, device)
        split = "val"
        out = {}
        for metric in evaluator.metric_names():
            out[f"{split}/{metric}/avg"] = \
                torch.tensor([scores[metric] for f_id, scores in score_dict_by_name.items() if f_id != 0]).mean().item()
            for f_id, scores in score_dict_by_name.items():
                out[f"{split}/{metric}/{f_id}"] = scores[metric]
        if self.logger is not None:
           self.logger.log(out, self.step)
        model_model = get_model_instance(model)
        model_model.set_train()
