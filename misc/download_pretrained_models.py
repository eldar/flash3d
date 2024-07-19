from pathlib import Path
import argparse
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_path", type=str, default="exp/re10k_v2")
    args = parser.parse_args()
    local_dir = Path(args.out_path) / "checkpoints"
    local_dir.mkdir(parents=True, exist_ok=True)
    model_path = hf_hub_download(
        repo_id="einsafutdinov/flash3d", 
        filename="model_re10k_v2.pth",
        local_dir=str(local_dir)
    )


if __name__ == "__main__":
    main()
