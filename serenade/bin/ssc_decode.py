#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Decode with trained singing style conversion model."""


import argparse
import math
import time
import torch
import torch.nn as nn
import numpy as np
import yaml
import logging
import os
import soundfile as sf
import json
import glob

import serenade.models
from serenade.datasets.audio_mel_dataset import FeatsDataset
from serenade.vocoder import Vocoder
from serenade.utils.masking import make_non_pad_mask

from joblib import load
from serenade.utils import read_hdf5, write_hdf5
from tqdm import tqdm

_c4_hz = 440 * 2 ** (3 / 12 - 1)
_c4_cent = 4800

class F0Statistics(object):
    """F0 statistics class
    Estimate F0 statistics and convert F0
    """

    def __init__(self):
        pass

    def estimate(self, f0list):
        """Estimate F0 statistics from list of f0
        Parameters
        ---------
        f0list : list, shape('f0num')
            List of several F0 sequence
        Returns
        ---------
        f0stats : array, shape(`[mean, std]`)
            Values of mean and standard deviation for logarithmic F0
        """

        n_files = len(f0list)
        for i in range(n_files):
            f0 = f0list[i]
            nonzero_indices = np.nonzero(f0)
            if i == 0:
                f0s = np.log(f0[nonzero_indices])
            else:
                f0s = np.r_[f0s, np.log(f0[nonzero_indices])]

        f0stats = np.array([np.mean(f0s), np.std(f0s)])
        return f0stats

    def convert(self, f0, orgf0stats, tarf0stats):
        """Convert F0 based on F0 statistics
        Parameters
        ---------
        f0 : array, shape(`T`, `1`)
            Array of F0 sequence
        orgf0stats, shape (`[mean, std]`)
            Vector of mean and standard deviation of logarithmic F0 for original speaker
        tarf0stats, shape (`[mean, std]`)
            Vector of mean and standard deviation of logarithmic F0 for target speaker
        Returns
        ---------
        cvf0 : array, shape(`T`, `1`)
            Array of converted F0 sequence
        """

        # get length and dimension
        T = len(f0)

        # perform f0 conversion
        cvf0 = np.zeros(T)

        nonzero_indices = f0 > 0
        cvf0[nonzero_indices] = np.exp(
            (tarf0stats[1] / orgf0stats[1])
            * (np.log(f0[nonzero_indices]) - orgf0stats[0])
            + tarf0stats[0]
        )

        return cvf0


def hz_to_cent_based_c4(hz):
    """Convert Hz to cent based on C4

    Args:
        hz (np.ndarray): array of Hz

    Returns:
        np.ndarray: array of cent
    """
    out = hz.copy()
    nonzero_indices = np.where(hz > 0)[0]
    out[nonzero_indices] = (
        1200 * np.log(hz[nonzero_indices] / _c4_hz) / np.log(2) + _c4_cent
    )
    return out


def cent_to_hz_based_c4(cent):
    """Convert cent to Hz based on C4

    Args:
        cent (np.ndarray): array of cent

    Returns:
        np.ndarray: array of Hz
    """
    out = cent.copy()
    nonzero_indices = np.where(cent > 0)[0]
    out[nonzero_indices] = (
        np.exp((cent[nonzero_indices] - _c4_cent) * np.log(2) / 1200) * _c4_hz
    )
    return out


def linear_midi_shift(sm, tm):
    f0class = F0Statistics()
    idx_s = sm > 0
    idx_t = tm > 0

    srcstats = f0class.estimate([sm])
    trgstats = f0class.estimate([tm])

    src_mean_cent = 1200 * np.log(np.exp(srcstats[0]) / _c4_hz) / np.log(2) +_c4_cent
    tgt_mean_cent = 1200 * np.log(np.exp(trgstats[0]) / _c4_hz) / np.log(2) +_c4_cent

    # round in semi-tone
    if (tgt_mean_cent - src_mean_cent) >= 0:
        shift = round((tgt_mean_cent - src_mean_cent) * 1.4 / 100) * 100
    else:
        shift = round((tgt_mean_cent - src_mean_cent) * (5/7) / 100) * 100

    sm[idx_s] = hz_to_cent_based_c4(sm[idx_s])
    sm[idx_s] = np.maximum(0, sm[idx_s] + shift)
    sm[idx_s] = cent_to_hz_based_c4(sm[idx_s])

    return sm


def get_random_ref_style(dumpdir, utt_id):
    filepath = os.path.join(dumpdir, f"{utt_id}.h5")
    ref_dict = {}
    dirname = os.path.dirname(filepath)
    ln, spk = os.path.basename(filepath).split(".")[0].split("_")[:2]

    styles = ["Breathy", "Falsetto", "Pharyngeal", "Mixed_Voice"]
    for style in styles:
        # Create glob pattern to match style files
        pattern = os.path.join(dirname, f"{ln}_{spk}_*_{style}_Group_*.h5")
        style_files = glob.glob(pattern)

        if not style_files:
            # NOTE: this probably breaks if you have more than 2 jobs for extracting files
            if "dump.2" in dirname:
                pattern = os.path.join(dirname.replace("dump.2", "dump.1"), f"{ln}_{spk}_*_{style}_Group_*.h5")
            elif "dump.1" in dirname:
                pattern = os.path.join(dirname.replace("dump.1", "dump.2"), f"{ln}_{spk}_*_{style}_Group_*.h5")
            style_files = glob.glob(pattern)

        if style_files:
            ref_dict[style] = np.random.choice(style_files)

    logging.info(f"Using reference styles: {ref_dict}")
    return ref_dict


def main():
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description=(
            "Decode with trained SSC model " "(See detail in bin/ssc_decode.py)."
        )
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help=(
            "yaml format configuration file. if not explicitly provided, "
            "it will be searched in the checkpoint directory. (default=None)"
        ),
    )
    parser.add_argument(
        "--feats-scp",
        "--scp",
        default=None,
        type=str,
        help=(
            "kaldi-style feats.scp file. "
            "you need to specify either feats-scp or dumpdir."
        ),
    )
    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        help=(
            "directory including feature files. "
            "you need to specify either feats-scp or dumpdir."
        ),
    )
    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="stats file for target denormalization.",
    )
    parser.add_argument(
        "--ref-dict",
        type=str,
        default=None,
        help="yaml format file containing reference styles.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save generated speech.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint file to be loaded.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load config
    if args.config is None:
        dirname = os.path.dirname(args.checkpoint)
        args.config = os.path.join(dirname, "config.yml")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # Read reference json file
    if args.ref_dict is not None:
        with open(args.ref_dict, "r") as f:
            ref_dict = json.load(f)
    else:
        logging.info("No reference dictionary provided, using random reference styles.")
        ref_dict = None

    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # load target stats for denormalization
    scaler = load(args.stats)
    config["trg_stats"] = {
        "mean": scaler["logmel"].mean_,
        "scale": scaler["logmel"].scale_,
    }

    # load vocoder
    vocoder = Vocoder(
        config["vocoder"]["checkpoint"],
        config["vocoder"]["config"],
        config["vocoder"]["stats"],
        device,
        trg_stats=config[
            "trg_stats"
        ],  # this is used to denormalized the converted features,
    )

    # check arguments
    if (args.feats_scp is not None and args.dumpdir is not None) or (
        args.feats_scp is None and args.dumpdir is None
    ):
        raise ValueError("Please specify either --dumpdir or --feats-scp.")

    # get dataset
    dataset = FeatsDataset(
        root_dir=args.dumpdir,
        scaler=scaler,
        score_type="est_lf0_score", # this is fixed, we never use gt_lf0_score in inference
        return_utt_id=True,
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )
    logging.info(f"The number of features to be decoded = {len(dataset)}.")

    # get model and load parameters
    model_class = getattr(serenade.models, config["model_type"])
    model = model_class(**config["model_params"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["model"])
    model = model.eval().to(device)
    logging.info(f"Loaded model parameters from {args.checkpoint}.")

    os.makedirs(args.outdir, exist_ok=True)

    # start generation
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, batch in enumerate(pbar, 1):
            start_time = time.time()

            utt_id = batch["utt_id"]
            logging.info(f"Decoding {utt_id}")

            x = batch["hubert"]
            score = batch["score"]
            logmel = batch["logmel"]
            lft = batch["loud"]
            lf0 = batch["lf0"]
            wave = batch["audio"]

            # save ground truth for easy reference
            sf.write(
                os.path.join(args.outdir, f"{utt_id}_gt.wav"),
                wave,
                config["sampling_rate"],
                "PCM_16",
            )

            x = torch.tensor(x, dtype=torch.float).to(device).unsqueeze(0)
            lengths = torch.tensor([x.shape[1]], dtype=torch.long).to(device)
            scores = torch.tensor(score, dtype=torch.float).to(device).unsqueeze(0)
            logmels = torch.tensor(logmel, dtype=torch.float).to(device).unsqueeze(0)
            lf0s = torch.tensor(lf0, dtype=torch.float).to(device).unsqueeze(0)
            lfts = torch.tensor(lft, dtype=torch.float).to(device).unsqueeze(0)

            if ref_dict is None:
                ref_dict = get_random_ref_style(args.dumpdir, utt_id)

            for key, ref_h5path in tqdm(ref_dict.items(), desc="Processing reference styles"):
                if key in utt_id:
                    # avoid reconstruction
                    continue

                logging.info(f"Processing reference style: {key}")
                #spk, style = key.split("_")
                style = key
                ref_cvec = read_hdf5(ref_h5path, "hubert")
                ref_mel = read_hdf5(ref_h5path, "logmel")
                ref_lft = read_hdf5(ref_h5path, "loud")
                ref_wave = read_hdf5(ref_h5path, "wave")
                ref_score = read_hdf5(ref_h5path, "est_lf0_score")
                ref_lf0 = read_hdf5(ref_h5path, "f0")

                sf.write(
                    os.path.join(args.outdir, f"00_{style}_reference.wav"),
                    ref_wave,
                    config["sampling_rate"],
                    "PCM_16",
                ) 

                # normalize 
                ref_cvec = (ref_cvec - scaler["hubert"].mean_) / scaler["hubert"].scale_
                ref_mel = (ref_mel - scaler["logmel"].mean_) / scaler["logmel"].scale_
                ref_cvec = torch.from_numpy(ref_cvec).float().to(device).unsqueeze(0)
                ref_mel = torch.from_numpy(ref_mel).float().to(device).unsqueeze(0)
                ref_lns = torch.tensor([ref_cvec.size(1)], dtype=torch.long, device=device)

                ref_score = (ref_score - scaler["score"].data_min_) / (scaler["score"].data_max_ - scaler["score"].data_min_)
                ref_score = torch.from_numpy(ref_score).float().to(device).view(1, -1, 1)
                
                ref_lft = (ref_lft - scaler["loud"].data_min_) / (scaler["loud"].data_max_ - scaler["loud"].data_min_)
                ref_lft = torch.from_numpy(ref_lft).float().to(device).unsqueeze(0)

                # shifted source lf0
                shifted_lf0 = linear_midi_shift(lf0, ref_lf0)

                # with reference style
                with torch.no_grad():
                    mel_ = model.inference(
                        x,
                        lengths,
                        scores,
                        lfts,
                        ref_cvec,
                        ref_lns,
                        ref_mel,
                        ref_score,
                        ref_lft,
                    )

                # ground truth
                with torch.no_grad():
                    wave, sr = vocoder.decode(mel_.squeeze(0))

                outname = f"{utt_id}_{style}"
                write_hdf5(
                    os.path.join(args.outdir, f"{outname}.h5"),
                    "lf0",
                    shifted_lf0.astype(np.float32),
                )
                sf.write(
                    os.path.join(args.outdir, f"{outname}.wav"),
                    wave.cpu().numpy(),
                    config["sampling_rate"],
                    "PCM_16",
                )


if __name__ == "__main__":
    main()
