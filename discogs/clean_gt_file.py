from argparse import ArgumentParser
import pickle
from pathlib import Path
import numpy as np


def clean_gt_file(input_file, output_file, data_base, n_bands, min_frames):
    with open(input_file, "rb") as i_file:
        data = pickle.load(i_file)

    new_data = dict()
    for key, value in data.items():
        # check key exists
        path = Path(data_base, key)
        key_ok, value_ok = False, False

        if path.exists():
            n_frames = path.stat().st_size // (2 * n_bands)  # each float16 has 2 bytes
            if n_frames > min_frames:
                key_ok = True

        # check value has at least one label
        if np.sum(value):
            value_ok = True

        if key_ok and value_ok:
            new_data[key] = value

    print(f"kept {len(new_data)} out of {len(data)} samples")

    with open(output_file, "wb") as o_file:
        pickle.dump(new_data, o_file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("--data-base")
    parser.add_argument("--n-bands", default=96, type=int)
    parser.add_argument("--min-frames", default=625, type=int)

    args = parser.parse_args()

    clean_gt_file(args.input_file, args.output_file, args.data_base, args.n_bands, args.min_frames)
