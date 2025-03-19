# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Analysis-synthesis script.

Analysis: WORLD vocoder
Synthesis: Pre-trained neural vocoder

"""

# A logger for this file
import copy
import os
from logging import getLogger

import glob
import h5py
from pathlib import Path
from tqdm import tqdm
import hydra
import librosa
import numpy as np
import pysptk
import pyworld as pw
import soundfile as sf
import torch
from hydra.utils import instantiate, to_absolute_path
from joblib import load
from omegaconf import DictConfig
from scipy.interpolate import interp1d
from sifigan.utils.features import SignalGenerator, dilated_factor
from serenade.utils import read_hdf5

logger = getLogger(__name__)

# All-pass-filter coefficients {key -> sampling rate : value -> coefficient}
ALPHA = {
    8000: 0.312,
    12000: 0.369,
    16000: 0.410,
    22050: 0.455,
    24000: 0.466,
    32000: 0.504,
    44100: 0.544,
    48000: 0.554,
}


def convert_continuos_f0(f0):
    # get uv information as binary
    uv = np.float32(f0 != 0)
    # get start and end of f0
    if (f0 == 0).all():
        logger.warn("all of the f0 values are 0.")
        return uv, f0, False
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]
    # padding start and end of f0 sequence
    cont_f0 = copy.deepcopy(f0)
    start_idx = np.where(cont_f0 == start_f0)[0][0]
    end_idx = np.where(cont_f0 == end_f0)[0][-1]
    cont_f0[:start_idx] = start_f0
    cont_f0[end_idx:] = end_f0
    # get non-zero frame index
    nz_frames = np.where(cont_f0 != 0)[0]
    # perform linear interpolation
    f = interp1d(nz_frames, cont_f0[nz_frames])
    cont_f0 = f(np.arange(0, cont_f0.shape[0]))

    return uv, cont_f0, True


@torch.no_grad()
@hydra.main(
    version_base=None, config_path="sifigan_config", config_name="ssc_postprocessing"
)
def main(config: DictConfig) -> None:
    """Run analysis-synthesis process."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Synthesize on {device}.")

    # load pre-trained model from checkpoint file
    model = instantiate(config.generator)
    state_dict = torch.load(
        to_absolute_path(config.checkpoint_path), map_location="cpu"
    )
    model.load_state_dict(state_dict["model"]["generator"])
    logger.info(f"Loaded model parameters from {config.checkpoint_path}.")
    model.remove_weight_norm()
    model.eval().to(device)

    # get scaler
    scaler = load(config.stats)

    # get data processor
    signal_generator = SignalGenerator(
        sample_rate=config.sample_rate,
        hop_size=int(config.sample_rate * config.frame_period * 0.001),
        sine_amp=config.sine_amp,
        noise_amp=config.noise_amp,
        signal_types=config.signal_types,
    )

    logger.info(f"Processing {config.in_dir}")
    wav_paths = glob.glob(os.path.join(config.in_dir, "**", "*.wav"), recursive=True)
    wav_paths = [
        path for path in wav_paths if "_reference" not in path and "_gt" not in path
    ]

    for wav_file in wav_paths:
        logger.info(f"Start processing {wav_file}")
        # FIXME: hardcoded stuff
        if "Soprano" in wav_file:
            f0_floor = 261
            f0_ceil = 1046
        elif "Alto" in wav_file:
            f0_floor = 196
            f0_ceil = 700
        elif "Tenor" in wav_file:
            f0_floor = 130
            f0_ceil = 440
        elif "Bass" in wav_file:
            f0_floor = 82
            f0_ceil = 330
        else:
            logger.info("WARNING UNKNOWN VOICE TYPE")
            f0_floor = 80
            f0_ceil = 1100

        if os.path.splitext(wav_file)[1] != ".wav":
            continue

        # WORLD analysis
        h5_path = wav_file.replace(".wav", ".h5")
        x, sr = sf.read(to_absolute_path(Path(wav_file)))
        if sr != config.sample_rate:
            x = librosa.resample(x, orig_sr=sr, target_sr=config.sample_rate)
        f0_cvt, t = pw.harvest(
            x,
            config.sample_rate,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
            frame_period=config.frame_period,
        )
        f0_ = read_hdf5(to_absolute_path(Path(h5_path)), "lf0")
        if f0_ is None:
            print(f"No h5 file containing f0 found for {wav_file}")
            continue

        # Interpolate f0_ to match the size of f0__
        if len(f0_) != len(f0_cvt):
            x_orig = np.arange(len(f0_))
            x_new = np.linspace(0, len(f0_) - 1, len(f0_cvt))
            f0_ = np.interp(x_new, x_orig, f0_.ravel())
            f0_ = np.maximum(f0_, 0)

        f0_ = f0_.squeeze().astype(np.float64)
        sp = pw.cheaptrick(x, f0_, t, config.sample_rate)
        ap = pw.d4c(x, f0_, t, config.sample_rate)
        mcep = pysptk.sp2mc(sp, order=config.mcep_dim, alpha=ALPHA[config.sample_rate])
        mcap = pysptk.sp2mc(ap, order=config.mcap_dim, alpha=ALPHA[config.sample_rate])
        bap = pw.code_aperiodicity(ap, config.sample_rate)

        # prepare f0 related features
        uv, cf0_, flag = convert_continuos_f0(f0_)
        cf0_ = cf0_.astype(np.float64)
        if not flag:
            continue

        uv = uv[:, np.newaxis]  # (T, 1)
        f0 = f0_[:, np.newaxis]  # (T, 1)
        cf0 = cf0_[:, np.newaxis]  # (T, 1)

        # prepare input acoustic features
        c = []
        for feat_type in config.aux_feats:
            if feat_type == "f0":
                c += [scaler[feat_type].transform(f0)]
            elif feat_type == "cf0":
                c += [scaler[feat_type].transform(cf0)]
            elif feat_type == "uv":
                c += [scaler[feat_type].transform(uv)]
            elif feat_type == "mcep":
                c += [scaler[feat_type].transform(mcep)]
            elif feat_type == "mcap":
                c += [scaler[feat_type].transform(mcap)]
            elif feat_type == "bap":
                c += [scaler[feat_type].transform(bap)]
        c = np.concatenate(c, axis=1)

        # prepare dense factors
        dfs = []
        for df, us in zip(
            config.dense_factors,
            np.cumprod(config.generator.upsample_scales),
        ):
            dfs += [
                np.repeat(dilated_factor(cf0, config.sample_rate, df), us)
                if config.df_f0_type == "cf0"
                else np.repeat(dilated_factor(f0, config.sample_rate, df), us)
            ]

        # convert to torch tensors
        f0 = torch.FloatTensor(f0).view(1, 1, -1).to(device)
        cf0 = torch.FloatTensor(cf0).view(1, 1, -1).to(device)
        c = torch.FloatTensor(c).unsqueeze(0).transpose(2, 1).to(device)
        dfs = [torch.FloatTensor(np.array(df)).view(1, 1, -1).to(device) for df in dfs]

        # generate input signals
        if config.sine_f0_type == "cf0":
            in_signal = signal_generator(cf0)
        elif config.sine_f0_type == "f0":
            in_signal = signal_generator(f0)

        # synthesize with the neural vocoder
        c = c.to(torch.float32)
        in_signal = in_signal.to(torch.float32)
        y = model(in_signal, c, dfs)[0]

        # save output signal as PCM 16 bit wav file
        # base = os.path.basename(wav)
        out_path = wav_file.replace(".wav", f"_sifigan.wav")
        sf.write(
            to_absolute_path(out_path),
            y.view(-1).cpu().numpy(),
            config.sample_rate,
            "PCM_16",
        )


if __name__ == "__main__":
    main()
