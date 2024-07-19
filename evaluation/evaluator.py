import math
import torch
import torch.nn as nn

from torchmetrics.image import \
    LearnedPerceptualImagePatchSimilarity, \
    StructuralSimilarityIndexMeasure


class Evaluator(nn.Module):
    def __init__(self, crop_border=True):
        super().__init__()

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr = lambda pred, gt: -10 * torch.log10(
            torch.mean((pred - gt) ** 2, dim=[1, 2, 3])
        ).mean()

        self.crop_border = crop_border
        self.metrics = {
            "ssim": self.ssim,
            "psnr": self.psnr,
            "lpips": lambda pred, gt: self.lpips(self.norm(pred), self.norm(gt))
        }

    @staticmethod
    def norm(img):
        return (img * 2 - 1).clamp(-1, 1)

    def metric_names(self):
        return self.metrics.keys()
    
    def forward(self, img_pred, img_gt):
        b, c, h, w = img_gt.shape

        if self.crop_border:
            margin = 0.05
            y0 = int(math.ceil(margin * h))
            y1 = int(math.floor((1-margin) * h))
            x0 = int(math.ceil(margin * w))
            x1 = int(math.floor((1-margin) * w))
            img_gt = img_gt[..., y0:y1, x0:x1]
            img_pred = img_pred[..., y0:y1, x0:x1]

        return {
            name: func(img_pred, img_gt).cpu().item() \
                for name, func in self.metrics.items()
        }
