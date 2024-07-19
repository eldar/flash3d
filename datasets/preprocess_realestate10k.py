from pathlib import Path
import gzip
import pickle
import argparse
from tqdm import tqdm

from datasets.re10k import load_seq_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", type=str)
    parser.add_argument("-d", "--data_path", type=str)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    split = args.split
    seq_data = load_seq_data(data_path, split)
    seq_keys = list(seq_data.keys())
    for seq in tqdm(seq_keys):
        if not data_path.joinpath(split, seq).is_dir():
            print(f"missing sequence {seq}")
            del seq_data[seq]

    file_path = data_path / f"{split}.pickle.gz"
    with gzip.open(file_path, "wb") as f:
        pickle.dump(seq_data, f)


if __name__ == "__main__":
    main()
