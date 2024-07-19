import torch
import torch.nn as nn
import numpy as np


def get_splits_and_inits(cfg):
    split_dimensions = []
    scale_inits = []
    bias_inits = []

    for g_idx in range(cfg.model.gaussians_per_pixel):
        if cfg.model.predict_offset:
            split_dimensions += [3]
            scale_inits += [cfg.model.xyz_scale]
            bias_inits += [cfg.model.xyz_bias]

        split_dimensions += [1, 3, 4, 3]
        scale_inits += [cfg.model.opacity_scale, 
                        cfg.model.scale_scale,
                        1.0,
                        5.0]
        bias_inits += [cfg.model.opacity_bias,
                        np.log(cfg.model.scale_bias),
                        0.0,
                        0.0]

        if cfg.model.max_sh_degree != 0:
            sh_num = (cfg.model.max_sh_degree + 1) ** 2 - 1
            sh_num_rgb = sh_num * 3
            split_dimensions.append(sh_num_rgb)
            scale_inits.append(cfg.model.sh_scale)
            bias_inits.append(0.0)
        if not cfg.model.one_gauss_decoder:
            break

    return split_dimensions, scale_inits, bias_inits, 


class GaussianDecoder(nn.Module):
    """transfer the predicted features into the gaussian parameters"""
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.scaling_lambda = cfg.model.scale_lambda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, split_dimensions=[3,1,3,4,3,9]):
        """split the gaussian parameters (default)
        offset: 3 channel
        opacity: 1 channel
        scale: 3 channel
        rotation: 4 channel
        color: 3 channel
        sh_feature: 9 channel
        """
        outputs = x.split(split_dimensions, dim=1)
        
        offset_list = []
        opacity_list = []
        scaling_list = []
        rotation_list = []
        feat_dc_list = []
        feat_rest_list = []

        for i in range(self.cfg.model.gaussians_per_pixel):
            if self.cfg.model.predict_offset and self.cfg.model.max_sh_degree != 0:
                offset_s, opacity_s, scaling_s, rotation_s, feat_dc_s, features_rest_s = outputs[i*6:(i+1)*6]
                offset_list.append(offset_s[:, None, ...])
                feat_rest_list.append(features_rest_s[:, None, ...])
            elif self.cfg.model.predict_offset:
                offset_s, opacity_s, scaling_s, rotation_s, feat_dc_s = outputs[i*5:(i+1)*5]
                offset_list.append(offset_s[:, None, ...])
            elif self.cfg.model.max_sh_degree != 0:
                opacity_s, scaling_s, rotation_s, feat_dc_s, features_rest_s = outputs[i*5:(i+1)*5]
                feat_rest_list.append(features_rest_s[:, None, ...])
            else:
                opacity_s, scaling_s, rotation_s, feat_dc_s = outputs[i*4:(i+1)*4]
            opacity_list.append(opacity_s[:, None, ...])
            scaling_list.append(scaling_s[:, None, ...])
            rotation_list.append(rotation_s[:, None, ...])
            feat_dc_list.append(feat_dc_s[:, None, ...])
            if not self.cfg.model.one_gauss_decoder:
                break
        
        # squeezing will remove dimension if there is only one gaussian per pixel
        opacity = torch.cat(opacity_list, dim=1)
        scaling = torch.cat(scaling_list, dim=1)
        rotation = torch.cat(rotation_list, dim=1)
        feat_dc = torch.cat(feat_dc_list, dim=1)

        out = {
            "gauss_opacity": self.opacity_activation(opacity),
            "gauss_scaling": self.scaling_activation(scaling) * self.scaling_lambda,
            "gauss_rotation": self.rotation_activation(rotation, dim=-3),
            "gauss_features_dc": feat_dc,
        }

        if self.cfg.model.predict_offset:
            offset = torch.cat(offset_list, dim=1)
            out["gauss_offset"] = offset
        if self.cfg.model.max_sh_degree != 0:
            features_rest = torch.cat(feat_rest_list, dim=1)
            out["gauss_features_rest"] = features_rest

        return out