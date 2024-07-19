import os
import json
import hydra
import torch
import numpy as np

from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from matplotlib import pyplot as plt
import torchvision.transforms.functional as TF

from models.model import GaussianPredictor, to_device
from evaluation.evaluator import Evaluator
from datasets.util import create_datasets
from misc.util import add_source_frame_id
from misc.visualise_3d import save_ply


def get_model_instance(model):
    """
    unwraps model from EMA object
    """
    return model.ema_model if type(model).__name__ == "EMA" else model

def evaluate(model, cfg, evaluator, dataloader, device=None, save_vis=False):
    model_model = get_model_instance(model)
    model_model.set_eval()

    score_dict = {}
    match cfg.dataset.name:
        case "re10k" | "nyuv2":
            # override the frame idxs used for eval
            target_frame_ids = [1, 2, 3]
            eval_frames = ["src", "tgt5", "tgt10", "tgt_rand"]
            for fid, target_name in zip(add_source_frame_id(target_frame_ids),
                                        eval_frames):
                score_dict[fid] = { "ssim": [],
                                    "psnr": [],
                                    "lpips": [],
                                    "name": target_name }
        case "kitti":
            if cfg.dataset.stereo:
                eval_frames = ["s0"]
                target_frame_ids = ["s0"]
                all_frames = add_source_frame_id(eval_frames)
            else:
                eval_frames = [1, 2]
                target_frame_ids = eval_frames
                all_frames = add_source_frame_id(target_frame_ids)
            for fid in all_frames:
                score_dict[fid] = { "ssim": [],
                                    "psnr": [],
                                    "lpips": [],
                                    "name": fid}

    dataloader_iter = iter(dataloader)
    for k in tqdm([i for i in range(len(dataloader.dataset) // cfg.data_loader.batch_size)]):

        if save_vis:
            out_dir = Path("/work/cxzheng/3D/splatvideo/eldar/visual_results/images")
            out_dir.mkdir(exist_ok=True)
            print(f"saving images to: {out_dir.resolve()}")
            seq_name = dataloader.dataset._seq_keys[k]
            out_out_dir = out_dir / seq_name
            out_out_dir.mkdir(exist_ok=True)
            out_pred_dir = out_out_dir / f"pred"
            out_pred_dir.mkdir(exist_ok=True)
            out_gt_dir = out_out_dir / f"gt"
            out_gt_dir.mkdir(exist_ok=True)
            out_dir_ply = out_out_dir / "ply"
            out_dir_ply.mkdir(exist_ok=True)

        try:
            inputs = next(dataloader_iter)
        except Exception as e:
            if cfg.dataset.name=="re10k":
                if cfg.dataset.test_split in ["pixelsplat_ctx1",
                                              "pixelsplat_ctx2",
                                              "latentsplat_ctx1",
                                              "latentsplat_ctx2"]:
                    print("Failed to read example {}".format(k))
                    continue
            raise e
        
        with torch.no_grad():
            if device is not None:
                to_device(inputs, device)
            inputs["target_frame_ids"] = target_frame_ids
            outputs = model(inputs)

        for f_id in score_dict.keys():
            pred = outputs[('color_gauss', f_id, 0)]
            if cfg.dataset.name == "dtu":
                gt = inputs[('color_orig_res', f_id, 0)]
                pred = TF.resize(pred, gt.shape[-2:])
            else:
                gt = inputs[('color', f_id, 0)]
            # should work in for B>1, however be careful of reduction
            out = evaluator(pred, gt)
            if save_vis:
                save_ply(outputs, out_dir_ply / f"{f_id}.ply", gaussians_per_pixel=model.cfg.model.gaussians_per_pixel)
                pred = pred[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
                gt = gt[0].clip(0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()
                plt.imsave(str(out_pred_dir / f"{f_id:03}.png"), pred)
                plt.imsave(str(out_gt_dir / f"{f_id:03}.png"), gt)
            for metric_name, v in out.items():
                score_dict[f_id][metric_name].append(v)

    metric_names = ["psnr", "ssim", "lpips"]
    score_dict_by_name = {}
    for f_id in score_dict.keys():
        score_dict_by_name[score_dict[f_id]["name"]] = {}
        for metric_name in metric_names:
            # compute mean
            score_dict[f_id][metric_name] = sum(score_dict[f_id][metric_name]) / len(score_dict[f_id][metric_name])
            # original dict has frame ids as integers, for json out dict we want to change them
            # to the meaningful names stored in dict
            score_dict_by_name[score_dict[f_id]["name"]][metric_name] = score_dict[f_id][metric_name]

    for metric in metric_names:
        vals = [score_dict_by_name[f_id][metric] for f_id in eval_frames]
        print(f"{metric}:", np.mean(np.array(vals)))

    return score_dict_by_name


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None
)
def main(cfg: DictConfig):
    print("current directory:", os.getcwd())
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg['runtime']['output_dir']
    os.chdir(output_dir)
    print("Working dir:", output_dir)

    cfg.data_loader.batch_size = 1
    cfg.data_loader.num_workers = 1
    model = GaussianPredictor(cfg)
    device = torch.device("cuda:0")
    model.to(device)
    if (ckpt_dir := model.checkpoint_dir()).exists():
        # resume training
        model.load_model(ckpt_dir, ckpt_ids=0)
    
    evaluator = Evaluator(crop_border=cfg.dataset.crop_border)
    evaluator.to(device)

    split = "test"
    save_vis = cfg.eval.save_vis
    dataset, dataloader = create_datasets(cfg, split=split)
    score_dict_by_name = evaluate(model, cfg, evaluator, dataloader, 
                                  device=device, save_vis=save_vis)
    print(json.dumps(score_dict_by_name, indent=4))
    if cfg.dataset.name=="re10k":
        with open("metrics_{}_{}_{}.json".format(cfg.dataset.name, split, cfg.dataset.test_split), "w") as f:
            json.dump(score_dict_by_name, f, indent=4)
    with open("metrics_{}_{}.json".format(cfg.dataset.name, split), "w") as f:
        json.dump(score_dict_by_name, f, indent=4)
    

if __name__ == "__main__":
    main()
