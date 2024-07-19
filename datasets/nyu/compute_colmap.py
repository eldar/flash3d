import math
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from imageio.v3 import imwrite, imread
import cv2

from datasets.colmap_database import COLMAPDatabase
from datasets.colmap_utils import rotmat2qvec, read_model
from datasets.colmap_misc import is_computed
from datasets.nyu.camera import camera_params, make_K
from slurm import p40_executor


DATA_PATH = Path("/scratch/shared/nfs1/eldar/data/nyuv2")


def get_sub_dirs(dir_colmap):
    database_path = dir_colmap / "database.db"
    dir_images = dir_colmap / "images"
    dir_cameras = dir_colmap / "sparse_input"
    return database_path, dir_images, dir_cameras


def get_sub_dirs(dir_colmap):
    database_path = dir_colmap / "database.db"
    dir_images = dir_colmap / "images"
    dir_cameras = dir_colmap / "sparse_input"
    return database_path, dir_images, dir_cameras


def get_nyu_test_sequences():
    labeled_dir = Path("/scratch/shared/nfs1/eldar/data/nyuv2_labeled")
    list_fn = labeled_dir / "test_sequences.txt"
    with open(list_fn) as f:
        sequences = [l.rstrip() for l in f.readlines()]
    return sequences


def run_colmap(dir_colmap, cuda_device):
    db_path, dir_images, dir_cameras = get_sub_dirs(dir_colmap)
    print("db_path", db_path)
    print("dir_images", dir_images)
    dir_out = dir_colmap / "sparse"
    dir_out.mkdir(exist_ok=True)

    colmap_exe = "/work/eldar/apps/colmap/colmap"

    env_vars = os.environ.copy()
    env_vars |= {
        "LD_LIBRARY_PATH": "/work/yashsb/apps/cuda-11.8/lib64:/work/eldar/apps/colmap",
        "CUDA_VISIBLE_DEVICES": cuda_device
    }

    logfile = open(dir_colmap / "log.txt", "w")

    print("Running feature extractor")
    logfile.write("Running feature extractor\n")
    logfile.flush()
    out = subprocess.run(
        [
            colmap_exe, "feature_extractor",
            "--database_path", db_path,
            "--image_path", dir_images
        ],
        env=env_vars,
        capture_output=True
    )
    log = f"STDOUT:\n{out.stdout.decode('utf-8')}\n\nSTDERR:\n{out.stderr.decode('utf-8')}\n"
    print(log)
    logfile.write(log)
    logfile.flush()

    msg = "Running exhaustive matcher"
    print(msg)
    logfile.write(f"{msg}\n")
    logfile.flush()
    out = subprocess.run(
        [
            colmap_exe, "exhaustive_matcher",
            "--database_path", db_path
        ],
        env=env_vars,
        capture_output=True
    )
    log = f"STDOUT:\n{out.stdout.decode('utf-8')}\n\nSTDERR:\n{out.stderr.decode('utf-8')}\n"
    print(log)
    logfile.write(log)
    logfile.flush()

    msg = "Running mapper"
    print(msg)
    logfile.write(f"{msg}\n")
    logfile.flush()

    out = subprocess.run(
        [
            colmap_exe, "mapper",
            "--database_path", db_path,
            "--image_path", dir_images,
            "--output_path", dir_out
        ],
        env=env_vars,
        capture_output=True
    )
    log = f"STDOUT:\n{out.stdout.decode('utf-8')}\n\nSTDERR:\n{out.stderr.decode('utf-8')}\n"
    print(log)
    logfile.write(log)

    logfile.close()


def add_camera(db):
    W = 640
    H = 480

    (fx, fy, cx, cy), (k1, k2, p1, p2, k3) = camera_params()

    rectified = True
    if rectified:
        # PINHOLE
        model = 1
        param_arr = np.array([fx, fy, cx, cy])
    else:
        # FULL_OPENCV
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        model = 6
        param_arr = np.array([fx, fy, cx, cy, k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])
    cam_id = db.add_camera(model, int(W), int(H), param_arr)

    distortion = np.array([k1, k2, p1, p2, k3])
    intr = make_K(fx, fy, cx, cy)

    return cam_id, intr, distortion


def compute_sfm(seq_name):
    data_path = DATA_PATH

    print("Sequence:", seq_name)

    seq_dir = data_path / seq_name
    if not seq_dir.exists():
        return

    out_root_dir = Path("/scratch/shared/nfs1/eldar/data/nyuv2_colmap")

    colmap_path = out_root_dir / seq_name
    if is_computed(colmap_path / "sparse/0"):
        print(f"Sequence {seq_name} already computed")
        return
    colmap_path.mkdir(exist_ok=True, parents=True)

    db_path, dir_images, dir_cameras = get_sub_dirs(colmap_path)
    db_path = colmap_path / "database.db"
    
    db_path.unlink(missing_ok=True)
    db = COLMAPDatabase.connect(db_path)
    db.create_tables()

    cam_id, intr, distortion = add_camera(db)

    image_dir = seq_dir
    frames = [f.name for f in sorted(list(image_dir.glob("*.ppm")))]

    out_image_dir = colmap_path / "images"
    out_image_dir.mkdir(exist_ok=True, parents=True)

    for frame in frames:
        img_src = image_dir / frame
        new_name = f"{Path(frame).stem}.jpg"
        img_dst = out_image_dir / new_name

        img = imread(img_src)
        img_un = cv2.undistort(img, intr, distortion)
        imwrite(img_dst, img_un, quality=100)

        image_id = db.add_image(new_name, cam_id)

    db.commit()
    db.close()

    cuda_device = "0"
    run_colmap(colmap_path, cuda_device)

    shutil.rmtree(out_image_dir, ignore_errors=True)


def main():
    sequences = get_nyu_test_sequences()
    sequences = [seq for seq in sequences if DATA_PATH.joinpath(seq).exists()]
    executor = p40_executor()
    jobs = executor.map_array(compute_sfm, sequences)


if __name__ == "__main__":
    main()