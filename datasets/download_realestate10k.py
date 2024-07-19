"""
Author: Felix Wimbauer
Source: https://github.com/Brummi/BehindTheScenes/blob/main/datasets/realestate10k/download_realestate10k.py
"""
import argparse
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path
from time import sleep

from pytubefix import YouTube
import tqdm
from subprocess import call


class Data:
    def __init__(self, url, seqname, list_timestamps):
        self.url = url
        self.list_seqnames = []
        self.list_list_timestamps = []

        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def add(self, seqname, list_timestamps):
        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def __len__(self):
        return len(self.list_seqnames)


def process(data, seq_id, videoname, output_root):
    seqname = data.list_seqnames[seq_id]
    out_path = output_root / seqname
    if not out_path.exists():
        out_path.mkdir(exist_ok=True, parents=True)
    else:
        print("[INFO] Something Wrong, stop process")
        return True

    list_str_timestamps = []
    for timestamp in data.list_list_timestamps[seq_id]:
        timestamp = int(timestamp / 1000)
        str_hour = str(int(timestamp / 3600000)).zfill(2)
        str_min = str(int(int(timestamp % 3600000) / 60000)).zfill(2)
        str_sec = str(int(int(int(timestamp % 3600000) % 60000) / 1000)).zfill(2)
        str_mill = str(int(int(int(timestamp % 3600000) % 60000) % 1000)).zfill(3)
        _str_timestamp = str_hour + ":" + str_min + ":" + str_sec + "." + str_mill
        list_str_timestamps.append(_str_timestamp)

    # extract frames from a video
    for idx, str_timestamp in enumerate(list_str_timestamps):
        call(("ffmpeg", "-ss", str_timestamp, "-i", str(videoname), "-vframes", "1", "-f", "image2", str(out_path / f'{data.list_list_timestamps[seq_id][idx]}.jpg')), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return False


def wrap_process(list_args):
    return process(*list_args)


class DataDownloader:
    def __init__(self, data_path: Path, out_path: Path, tmp_path: Path, mode='test'):
        print("[INFO] Loading data list ... ", end='')
        self.data_path = data_path
        self.out_path = out_path
        self.tmp_path = tmp_path
        self.mode = mode

        self.list_seqnames = sorted(self.data_path.glob('*.txt'))

        self.is_done = out_path.exists()

        out_path.mkdir(exist_ok=True, parents=True)

        self.list_data = {}
        for txt_file in tqdm.tqdm(self.list_seqnames):
            seq_name = txt_file.stem

            # extract info from txt
            with open(txt_file, "r") as seq_file:
                lines = seq_file.readlines()
                youtube_url = ""
                list_timestamps = []
                for idx, line in enumerate(lines):
                    if idx == 0:
                        youtube_url = line.strip()
                    else:
                        timestamp = int(line.split(' ')[0])
                        list_timestamps.append(timestamp)

            if youtube_url in self.list_data:
                self.list_data[youtube_url].add(seq_name, list_timestamps)
            else:
                self.list_data[youtube_url] = Data(youtube_url, seq_name, list_timestamps)

        print(" Done! ")
        print("[INFO] {} movies are used in {} mode".format(len(self.list_data), self.mode))

    def run(self):
        print("[INFO] Start downloading {} movies".format(len(self.list_data)))

        for global_count, data in enumerate(self.list_data.values()):
            print("[INFO] Downloading {} ".format(data.url))
            current_file = self.tmp_path / f"current_{self.mode}"

            call(("rm", "-r", str(current_file)))

            try:
                # sometimes this fails because of known issues of pytube and unknown factors
                yt = YouTube(data.url)
                stream = yt.streams.filter(res='360p').first()
                stream.download(str(current_file))
            except:
                with open(os.path.join(str(self.data_path.parent), 'failed_videos_' + self.mode + '.txt'), 'a') as f:
                    for seqname in data.list_seqnames:
                        f.writelines(seqname + '\n')
                continue

            sleep(1)

            current_file = next(current_file.iterdir())

            if len(data) == 1:  # len(data) is len(data.list_seqnames)
                process(data, 0, current_file, self.out_path)
            else:
                with Pool(processes=4) as pool:
                    pool.map(wrap_process, [(data, seq_id, current_file, self.out_path) for seq_id in range(len(data))])

            print(f"[INFO] Extracted {sum(map(len, data.list_list_timestamps))}")

            # remove videos
            call(("rm", str(current_file)))
            # os.system(command)

            if self.is_done:
                return False

        return True

    def show(self):
        print("########################################")
        global_count = 0
        for data in self.list_data.values():
            # print(" URL : {}".format(data.url))
            for idx in range(len(data)):
                # print(" SEQ_{} : {}".format(idx, data.list_seqnames[idx]))
                # print(" LEN_{} : {}".format(idx, len(data.list_list_timestamps[idx])))
                global_count = global_count + 1
            # print("----------------------------------------")

        print("TOTAL : {} sequnces".format(global_count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str)
    parser.add_argument("-d", "--data_path", type=str)
    parser.add_argument("-o", "--out_path", type=str)
    tmpdir = os.environ.get('TMPDIR')
    parser.add_argument("-t", "--tmp_path", default=tmpdir, type=str)

    args = parser.parse_args()
    mode = args.mode
    data_path = Path(args.data_path)
    out_path = Path(args.out_path)
    tmp_path = Path(args.tmp_path)

    if mode not in ["test", "train"]:
        raise ValueError(f"Invalid split mode: {mode}")

    data_path = data_path / mode
    out_path = out_path / mode
    downloader = DataDownloader(
        data_path=data_path,
        out_path=out_path,
        tmp_path=tmp_path,
        mode=mode)

    downloader.show()
    is_ok = downloader.run()

    if is_ok:
        print("Done!")
    else:
        print("Failed")


if __name__ == "__main__":
    main()


