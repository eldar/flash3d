import os
import shutil
import logging
import tarfile
from pathlib import Path

project_name = "monosplat"

def get_local_dir():
    tmp = os.environ["TMPDIR"] if "TMPDIR" in os.environ else "/tmp"
    if "SLURM_JOB_ID" in os.environ:
        sub_dir = f"{project_name}/{os.environ['SLURM_JOB_ID']}"
    else:
        sub_dir = project_name
    tmp = os.path.join(tmp, sub_dir)
    return Path(tmp)

def local_storage_path(filename):
    return get_local_dir() / Path(filename).name

def copy_to_local_storage(filename, rank=None):
    storage = get_local_dir()
    os.makedirs(storage, exist_ok=True)
    new_filename = local_storage_path(filename)
    filename = Path(filename)
    if rank is not None and rank != 0:
        return new_filename
    if not new_filename.is_file() or \
        filename.stat().st_size != new_filename.stat().st_size:
        logging.info(f"Copying {str(filename)} to {str(new_filename)} ...")
        shutil.copyfile(filename, new_filename)
        logging.info(f"Finished copying.")
    return new_filename

def extract_tar(fn, unzip_dir):
    unzip_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Unpacking {fn} to {unzip_dir} ...")
    tf = tarfile.open(fn)
    tf.extractall(unzip_dir, filter='fully_trusted')
    logging.info(f"Finished unpacking.")
    fn.unlink()
    logging.info(f"Deleted {str(fn)}.")