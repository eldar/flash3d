import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf
import neptune as neptune
from neptune.utils import stringify_unsupported

from misc.neptune_token import NEPTUNE_API_TOKEN, USER_NAME, PROJECT_NAME


def get_all_keys(d):
    for key, value in d.items():
        yield d, key
        if isinstance(value, dict):
            yield from get_all_keys(value)


def setup_logging_dir():
    tmp_dir = tempfile._get_default_tempdir()
    neptune_dir = Path(tmp_dir).joinpath(".neptune")
    neptune_dir.mkdir(exist_ok=True)
    target = Path(".neptune")
    if not target.exists():
        # os.unlink(target)
        os.symlink(neptune_dir, ".neptune")


class NeptuneLogger:
    def __init__(self, cfg):
        cfg_dict = OmegaConf.to_container(cfg)
        del cfg_dict["config"]
        setup_logging_dir()
        self.run = self._setup(cfg)
        for d, k in get_all_keys(cfg_dict):
            if d[k] is None:
                d[k] = "none"
            elif type(d[k]) is list:
                d[k] = stringify_unsupported(d[k])
        self.run["cfg"] = cfg_dict
        self.run["exp"] = cfg.config.exp_name

        if "SLURM_JOB_ID" in os.environ:
            SLURM_ID = os.environ['SLURM_JOB_ID']
            self.run["SLURM"] = SLURM_ID
            print(f"SLURM job ID: {SLURM_ID}")

    @staticmethod
    def _setup(cfg):
        CONNECTION_MODE = "debug" if cfg.run.debug else "async"
        with_id = None
        # sys_id_file = ".hydra/neptune.id"
        # if Path(sys_id_file).exists():
        #     with open(sys_id_file) as f:
        #         with_id = f.read()
        run = neptune.init_run(
            with_id=with_id,
            project=f"{USER_NAME}/{PROJECT_NAME}",
            api_token=NEPTUNE_API_TOKEN,
            name=cfg.config.exp_name,
            mode=CONNECTION_MODE
        )
        # if with_id is None:
        #     with open(sys_id_file, "w") as f:
        #         f.write(run["sys/id"].fetch())
        return run

    def log(self, values, step):
        for key, value in values.items():
            self.run[key].log(value, step=step)
    
    def log3d(self, kv, step):
        pass

    def upload_file(self, key, filename):
        self.run[key].upload(neptune.types.File(filename))
    
    def upload_image(self, key, image):
        self.run[key].upload(neptune.types.File.as_image(image))

    def log_image(self, key, image, step=0):
        self.run[key].append(neptune.types.File.as_image(image), step=step)


def setup_logger(cfg):
    return NeptuneLogger(cfg)
