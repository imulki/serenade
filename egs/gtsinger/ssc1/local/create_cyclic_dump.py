import glob
import os
import argparse

from tqdm import tqdm
from serenade.utils import read_hdf5, write_hdf5

"""
Creates a cyclic_logmel feature in the dump directory for cyclic training.
Basically just copies the source logmel feature from the source, making the conditioning features from converted but targets from ground truth.
"""

def create_cyclic_dump(outdir, dumpdir, train_set):

    h5_paths = glob.glob(os.path.join(outdir, "dump.*", "*.h5"))
    with open(os.path.join(outdir, "cyclic.log"), "w") as f:
        for h5_path in tqdm(h5_paths):
            f.write(f"Processing {h5_path}\n")

            orig_logmel = read_hdf5(h5_path, "logmel")
            style = h5_path.split("/")[-1].split("_")[-1][:-3]

            # get style from source dump directory
            if style in ["Pharyngeal", "Glissando", "Breathy", "Vibrato", "Falsetto", "Voice"]:
                if style == "Voice":
                    style = "Mixed_Voice"
                src_h5_basename = os.path.basename(h5_path).replace(f"_{style}", "")

                # finds the source file in the source dump directory
                src_h5_path = os.path.join(f"{dumpdir}/{train_set}/raw/dump.1", src_h5_basename)
                if not os.path.exists(src_h5_path):
                    src_h5_path = os.path.join(f"{dumpdir}/{train_set}/raw/dump.2", src_h5_basename)

                # get original logmel, rename it as cyclic_logmel
                cyclic_logmel = read_hdf5(src_h5_path, "logmel")
            else:
                # reconstruction of unconverted samples
                cyclic_logmel = orig_logmel

            # save as cyclic_logmel for targets during training
            if cyclic_logmel is None:
                print(f"Warning: {h5_path} is not found in {src_h5_path}")
                continue
            write_hdf5(h5_path, "cyclic_logmel", cyclic_logmel)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--dumpdir", type=str, required=True)
    parser.add_argument("--train_set", type=str, required=True)
    args = parser.parse_args()
    create_cyclic_dump(args.outdir, args.dumpdir, args.train_set)
