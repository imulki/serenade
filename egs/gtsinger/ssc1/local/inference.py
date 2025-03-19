# -*- coding: utf-8 -*-

# 2024 Lester Violeta (Nagoya University)

"""
Diffusion-based acoustic model implementation for voice conversion
References:
    - https://github.com/MoonInTheRiver/DiffSinger
    - https://github.com/nnsvs/nnsvs
"""

import math
import torch
import torch.nn as nn
import numpy as np
import logging
import os
import soundfile as sf
import json
import glob

from serenade.models.serenade import Serenade
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


def linear_midi_shift(srcmidi, trgmidi):
    f0class = F0Statistics()
    sm = np.zeros_like(srcmidi)
    tm = np.zeros_like(trgmidi)
    idx_s = srcmidi > 0
    idx_t = trgmidi > 0
    sm[idx_s] = np.exp(srcmidi[idx_s])
    tm[idx_t] = np.exp(trgmidi[idx_t])

    srcstats = f0class.estimate([sm])
    trgstats = f0class.estimate([tm])

    src_mean_cent = 1200 * np.log(np.exp(srcstats[0]) / _c4_hz) / np.log(2) + _c4_cent
    tgt_mean_cent = 1200 * np.log(np.exp(trgstats[0]) / _c4_hz) / np.log(2) + _c4_cent

    # round in semi-tone
    if (tgt_mean_cent - src_mean_cent) >= 0:
        shift = round((tgt_mean_cent - src_mean_cent) * 1.4 / 100) * 100
    else:
        shift = round((tgt_mean_cent - src_mean_cent) * (5 / 7) / 100) * 100

    sm[idx_s] = hz_to_cent_based_c4(sm[idx_s])
    sm[idx_s] = np.maximum(0, sm[idx_s] + shift)
    sm[idx_s] = cent_to_hz_based_c4(sm[idx_s])
    sm[idx_s] = np.log(sm[idx_s])

    return sm


score_type = "est_lf0_score"
exp_path = "/data/group1/z44568r/toolkits2/singstyle/egs/singstyle111/vc3/exp/train-gtsinger-cyclic_v9_proposed_cyclic_fix"

inference_dir = "results_est_1.0"
wav_paths = glob.glob("dump/test-gtsinger/raw/dump.*/*.h5")

# Read reference json file
with open("conf/refstyles.json", "r") as f:
    ref_dict = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the model
model = Serenade(
    input_dim=768,
    output_dim=80,
    encoder_channels=80,
    decoder_channels=512,
    decoder_attention_head_dim=512,
)

# Load model from checkpoint
checkpoint_path = os.path.join(exp_path, "checkpoint-200000steps.pkl")
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model"])
model = model.eval().to(device)

# Load stats for normalization/denormalization
stats_path = os.path.join(exp_path, "stats.joblib")
scaler = load(stats_path)

# load HiFiGAN
config = {}
config["trg_stats"] = {
    "mean": scaler["logmel"].mean_,
    "scale": scaler["logmel"].scale_,
}

hifigan_vocoder = Vocoder(
    "/data/group1/z44568r/toolkits2/ParallelWaveGAN/egs/m4singer/voc1/exp/train-gtsinger_pt300k/checkpoint-300000steps.pkl",
    "/data/group1/z44568r/toolkits2/ParallelWaveGAN/egs/m4singer/voc1/exp/train-gtsinger_pt300k/config.yml",
    "/data/group1/z44568r/toolkits2/ParallelWaveGAN/egs/m4singer/voc1/exp/train-gtsinger_pt300k/stats.h5",
    device,
    trg_stats=config[
        "trg_stats"
    ],  # this is used to denormalized the converted features,
)


for filepath in tqdm(wav_paths, desc="Processing audio files"):
    spk = os.path.basename(filepath).replace("English_EN-", "")
    spk = "_".join(spk.split("_")[:2])
    spk = spk.replace("-", "").lower().split("_")[0]

    x = read_hdf5(filepath, "hubert")
    lengths = torch.tensor([x.shape[0]]).to(device)
    scores = read_hdf5(filepath, score_type)
    lft = read_hdf5(filepath, "loud")
    targets = read_hdf5(filepath, "logmel")
    wave = read_hdf5(filepath, "wave")
    lf0 = read_hdf5(filepath, "f0")
    lf0 = lf0[:, np.newaxis].copy()
    nonzero_indices = np.nonzero(lf0)
    lf0[nonzero_indices] = np.log(lf0[nonzero_indices])
    vuv = read_hdf5(filepath, "vuv")
    lf0 = lf0 * vuv

    x = (x - scaler["hubert"].mean_) / scaler["hubert"].scale_
    x = torch.from_numpy(x).view(1, -1, 768).float().to(device)

    targets = (targets - scaler["logmel"].mean_) / scaler["logmel"].scale_
    targets = torch.from_numpy(targets).view(1, -1, 80).float().to(device)

    scores = (scores - scaler["lf0"].data_min_) / (
        scaler["lf0"].data_max_ - scaler["lf0"].data_min_
    )
    scores = torch.from_numpy(scores).view(1, -1, 1).float().to(device)

    lft = (lft - scaler["loud"].data_min_) / (
        scaler["loud"].data_max_ - scaler["loud"].data_min_
    )
    lft = torch.from_numpy(lft).view(1, -1, 1).float().to(device)

    sr = 24000

    os.makedirs(args.outdir, exist_ok=True)
    sf.write(
        os.path.join(args.outdir, f"{os.path.basename(filepath).split('.')[0]}_gt.wav"),
        wave,
        sr,
        "PCM_16",
    )

    for key, ref_h5path in tqdm(ref_dict.items(), desc="Processing reference styles"):
        if key in filepath:
            continue
        spk, style = key.split("_")

        ref_cvec = read_hdf5(ref_h5path, "hubert")
        ref_mel = read_hdf5(ref_h5path, "logmel")
        ref_lft = read_hdf5(ref_h5path, "loud")
        ref_wave = read_hdf5(ref_h5path, "wave")
        ref_score = read_hdf5(ref_h5path, score_type)
        ref_lf0 = read_hdf5(ref_h5path, "lf0")
        ref_vuv = read_hdf5(ref_h5path, "vuv")
        ref_lf0 = ref_lf0 * ref_vuv

        sf.write(
            os.path.join(outdir, f"00_{style}_reference.wav"),
            ref_wave,
            sr,
            "PCM_16",
        )

        # normalize
        ref_cvec = (ref_cvec - scaler["hubert"].mean_) / scaler["hubert"].scale_
        ref_mel = (ref_mel - scaler["logmel"].mean_) / scaler["logmel"].scale_
        ref_cvec = torch.from_numpy(ref_cvec).view(1, -1, 768).float().to(device)
        ref_mel = torch.from_numpy(ref_mel).view(1, -1, 80).float().to(device)
        ref_lns = torch.tensor([ref_cvec.size(1)], dtype=torch.long, device=device)

        ref_score = (ref_score - scaler["lf0"].data_min_) / (
            scaler["lf0"].data_max_ - scaler["lf0"].data_min_
        )
        ref_score = torch.from_numpy(ref_score).view(1, -1, 1).float().to(device)

        ref_lft = (ref_lft - scaler["loud"].data_min_) / (
            scaler["loud"].data_max_ - scaler["loud"].data_min_
        )
        ref_lft = torch.from_numpy(ref_lft).view(1, -1, 1).float().to(device)

        # shifted source lf0
        shifted_lf0 = linear_midi_shift(lf0, ref_lf0)

        # with reference style
        with torch.no_grad():
            mel_ = model.inference(
                x,
                lengths,
                scores,
                lft,
                ref_cvec,
                ref_lns,
                ref_mel,
                ref_score,
                ref_lft,
                src_logmel=targets,
            )

        # ground truth
        with torch.no_grad():
            wave, sr = vocoder.decode(mel_.squeeze(0))

        outname = f"{os.path.basename(filepath).split('.')[0]}_{spk}_{style}"
        write_hdf5(
            os.path.join(outdir, f"{outname}.h5"),
            "lf0",
            shifted_lf0.astype(np.float32),
        )
        sf.write(
            os.path.join(outdir, f"{outname}.wav"),
            wave.cpu().numpy(),
            sr,
            "PCM_16",
        )
