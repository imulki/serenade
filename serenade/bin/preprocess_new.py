#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Perform preprocessing and raw feature extraction."""

import argparse
import logging
import os
import json
import librosa
import mido
import pyworld
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

import yaml

from tqdm import tqdm

from serenade.datasets import AudioSCPDataset
from serenade.utils import write_hdf5
from serenade.modules.phoneme_midi.decoding import FramewiseDecoder
from serenade.modules.phoneme_midi.model import TranscriptionModel
from transformers import HubertModel

import torch
import torch.nn.functional as F
import random

from scipy.interpolate import UnivariateSpline

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
        self.final_proj = torch.nn.Linear(
            config.hidden_size, config.classifier_proj_size
        )


def read_and_resample_midi(midi_file_path, hop_length):
    mid = mido.MidiFile(midi_file_path)

    # Extract note events and track tempo changes
    note_events = []
    tempo = 500000  # Default tempo (120 BPM)
    current_time = 0
    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            if msg.type == "note_on" or msg.type == "note_off":
                note_events.append((current_time, msg.type, msg.note, msg.velocity))
            elif msg.type == "set_tempo":
                tempo = msg.tempo

    # Sort note events by time
    note_events.sort(key=lambda x: x[0])

    # Calculate the total duration of the MIDI file in seconds
    ticks_per_beat = mid.ticks_per_beat
    total_ticks = note_events[-1][0] if note_events else 0
    total_beats = total_ticks / ticks_per_beat
    total_duration_seconds = total_beats * (tempo / 1000000)

    # Create time points for resampling
    resampled_times = np.arange(0, total_duration_seconds, hop_length)

    # Initialize resampled data
    resampled_data = np.zeros((len(resampled_times), 128))

    # Track active notes
    active_notes = {}

    # Resample the MIDI data
    for event in note_events:
        time, event_type, note, velocity = event
        time_seconds = (time / ticks_per_beat) * (tempo / 1000000)
        index = int(time_seconds // hop_length)

        if event_type == "note_on" and velocity > 0:
            active_notes[note] = (index, velocity)
        elif event_type == "note_off" or (event_type == "note_on" and velocity == 0):
            if note in active_notes:
                start_index, start_velocity = active_notes[note]
                resampled_data[start_index : index + 1, note] = start_velocity
                del active_notes[note]

    # Handle any notes that are still active at the end of the file
    for note, (start_index, velocity) in active_notes.items():
        resampled_data[start_index:, note] = velocity

    # Convert resampled data to frequencies
    frequencies = np.zeros(len(resampled_times))
    midi = np.zeros(len(resampled_times))
    for i, time_slice in enumerate(resampled_data):
        active_notes = np.nonzero(time_slice)[0]
        if len(active_notes) > 0:
            highest_note = np.max(active_notes)
            frequencies[i] = librosa.midi_to_hz(highest_note)
            midi[i] = highest_note

    return resampled_times, resampled_data, frequencies, midi


def _midi_to_hz(x, log_f0=False):
    z = np.zeros_like(x)
    indices = x > 0
    z[indices] = librosa.midi_to_hz(x[indices])
    if log_f0:
        z[indices] = np.log(z[indices])
    return z


def loudness_extract(
    audio,
    sampling_rate,
    hop_length,
):
    stft = librosa.stft(audio, hop_length=hop_length)
    power_spectrum = np.square(np.abs(stft))
    bins = librosa.fft_frequencies(sr=sampling_rate)
    loudness = librosa.perceptual_weighting(power_spectrum, bins)
    loudness = librosa.db_to_amplitude(loudness)
    lft_noint = np.log(np.mean(loudness, axis=0) + 1e-5)

    return lft_noint


def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-10,
    log_base=10.0,
):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.
        log_base (float): Log base. If set to None, use np.log.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=fft_size,
        n_mels=num_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))

    if log_base is None:
        return np.log(mel)
    elif log_base == 10.0:
        return np.log10(mel)
    elif log_base == 2.0:
        return np.log2(mel)
    else:
        raise ValueError(f"{log_base} is not supported.")


def read_midi_json(note_seq, frame_shift):
    """Convert note sequence to frame-level MIDI array

    Args:
        note_seq: List of dicts containing note info
        frame_shift: Time between frames in seconds
    Returns:
        numpy array of MIDI values per frame
    """

    # Find total duration
    max_time = max([note["note_end"][-1] for note in note_seq])
    num_frames = int(np.ceil(max_time / frame_shift))

    # Initialize output array
    midi_frames = np.zeros(num_frames)

    # Fill in MIDI values frame by frame
    for note_dict in note_seq:
        notes = note_dict["note"]
        starts = note_dict["note_start"]
        ends = note_dict["note_end"]

        for note, start, end in zip(notes, starts, ends):
            start_frame = int(start / frame_shift)
            end_frame = int(end / frame_shift)
            midi_frames[start_frame:end_frame] = note

    return midi_frames


def midi_to_frames(midi_values, time_intervals, T, shift_ms=10):
    # Convert shift_ms to seconds
    shift_s = shift_ms / 1000.0

    # Calculate number of frames
    n_frames = int(np.ceil(T / shift_s))

    # Initialize output array
    frames = np.zeros(n_frames, dtype=np.int32)

    # For each note
    for midi, (start, end) in zip(midi_values, time_intervals):
        # Convert time to frame indices
        start_frame = int(np.floor(start / shift_s))
        end_frame = int(np.ceil(end / shift_s))

        # Ensure we don't exceed array bounds
        end_frame = min(end_frame, n_frames)

        # Fill in the frames where this note is active
        frames[start_frame:end_frame] = midi

    return np.array(frames, dtype=np.int32)


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess audio and then extract features (See detail in"
            " bin/preprocess.py)."
        )
    )
    parser.add_argument(
        "--wav-scp",
        "--scp",
        required=True,
        type=str,
        help="kaldi-style wav.scp file.",
    )
    parser.add_argument(
        "--segments",
        default=None,
        type=str,
        help=(
            "kaldi-style segments file. if use, you must to specify both scp and"
            " segments."
        ),
    )
    parser.add_argument(
        "--dumpdir",
        type=str,
        required=True,
        help="directory to dump feature files.",
    )
    parser.add_argument(
        "--midi-path",
        type=str,
        required=True,
        help="location of midi path",
    )
    parser.add_argument(
        "--f0-path",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--skip-gtmidi",
        type=bool,
        default=False,
        help="skip ground truth MIDI extraction.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
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

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    dataset = AudioSCPDataset(
        args.wav_scp,
        segments=args.segments,
        return_utt_id=True,
        return_sampling_rate=True,
    )

    # check directly existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ContentVec extraction
    hubert_extractor = HubertModelWithFinalProj.from_pretrained(
        "lengyue233/content-vec-best"
    )
    hubert_extractor = hubert_extractor.to(device).eval()
    hubert_extractor.config.apply_spec_augment = False
    hubert_extractor.feature_extractor.conv_layers[-1].conv.stride = (
        1,
    )  # convert to 10ms from 20ms frame shift

    # read MIDI file, TODO: hardcoded delimiter
    df = pd.read_csv(args.midi_path, delimiter=" /", names=["utt_id", "wav_path"])

    # predict MIDI from audio
    midi_model_file = config["midi_model_file"]
    ckpt = torch.load(midi_model_file)
    midi_config = ckpt["config"]
    model_state_dict = ckpt["model_state_dict"]

    midi_model = TranscriptionModel(midi_config)
    midi_model.load_state_dict(model_state_dict)
    midi_model.to(device)
    midi_model.eval()
    midi_decoder = FramewiseDecoder(midi_config)

    # read f0 min/max file
    with open(args.f0_path, "r") as file:
        f0_file = yaml.load(file, Loader=yaml.BaseLoader)

    # process each data
    for utt_id, (audio, fs) in tqdm(dataset):
        logging.info(f"processing {utt_id}")
        logging.info(f"raw audio shape: {audio.shape}")

        # skip if already exists
        # if os.path.exists(os.path.join(args.dumpdir, f"{utt_id}.h5")):
        #    logging.info(f"already exists, skipping {utt_id}")
        #    continue

        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        assert np.abs(audio).max() <= 1.0, (
            f"{utt_id} seems to be different from 16 bit PCM."
        )

        if fs != config["sampling_rate"]:
            logging.info(f"resampling {utt_id} from {fs} to {config['sampling_rate']}")
            audio = librosa.resample(
                audio,
                orig_sr=fs,
                target_sr=config["sampling_rate"],
            )
            fs = config["sampling_rate"]

        # trim silence
        # NOTE: if you want to use score MIDI, you should not trim silence
        if config["trim_silence"]:
            audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )

        logging.info(f"processed audio shape: {audio.shape}")
        # make sure the audio length and feature length are matched
        audio = np.pad(audio, (0, config["fft_size"]), mode="reflect")

        audio16k = librosa.resample(
            audio,
            orig_sr=config["sampling_rate"],
            target_sr=16000,
        )

        # logmel spectrogram
        logmel = logmelfilterbank(
            audio,
            sampling_rate=config["sampling_rate"],
            hop_size=config["hop_size"],
            fft_size=config["fft_size"],
            win_length=config["win_length"],
            window=config["window"],
            num_mels=config["num_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
            # keep compatibility
            log_base=config.get("log_base", 10.0),
        )  # [n_frames, n_dim]

        # Extract score MIDI information
        shiftms = config["hop_size"] * 1000 / config["sampling_rate"]
        # Read and resample MIDI
        midi_score_path = (
            df.loc[df["utt_id"] == utt_id, "wav_path"].iloc[0].replace(".wav", ".json")
        )
        midi_score_path = f"/{midi_score_path}"

        if not os.path.exists(midi_score_path) and not args.skip_gtmidi:
            logging.info(f"WARNING: {utt_id} has missing midi information")
            continue

        if not args.skip_gtmidi:
            with open(midi_score_path, "r") as f:
                note_seq = json.load(f)

            midi = read_midi_json(note_seq, shiftms / 1000)
            minf0 = max(63.5, min(librosa.midi_to_hz(midi - 6)))
            maxf0 = max(librosa.midi_to_hz(midi + 6))
            gt_lf0_score = _midi_to_hz(midi, log_f0=True)

        # extract loudness features using A-weighting
        loud = loudness_extract(audio, config["sampling_rate"], config["hop_size"])
        loud = np.expand_dims(loud, axis=-1)

        # extract F0 for post-processing
        spk_id = utt_id.split("_")[3].split("-")[1] # utt_id.split("_")[1].split("-")[1]
        try:
            minf0 = float(f0_file[spk_id]["minf0"])
            maxf0 = float(f0_file[spk_id]["maxf0"])
        except:
            # use default values for speech, f0 is not really needed for pretraining
            logging.info(f"cant find f0 extraction: {spk_id}")
            minf0 = 70
            maxf0 = 1100

        f0, t = pyworld.harvest(
            audio.astype(np.float64),
            fs=config["sampling_rate"],
            f0_floor=minf0,
            f0_ceil=maxf0,
            frame_period=config["shiftms"],
        )
        f0 = f0[:, np.newaxis].copy()
        vuv = (f0 != 0).astype(np.float32)

        x = torch.from_numpy(audio16k).view(1, -1).float().to(device)
        with torch.no_grad():
            # extract and resample contentvec (if necessary)
            raw_hubert = hubert_extractor(x)["last_hidden_state"].permute(0, 2, 1)
            frame_shift_int = int(config["sampling_rate"] * shiftms / 1000)
            scale_factor = (config["sampling_rate"] / frame_shift_int) * (160 / 16000)
            hubert = F.interpolate(raw_hubert, scale_factor=scale_factor)
            hubert = hubert.permute(0, 2, 1).squeeze(0).cpu().numpy()
            raw_hubert = raw_hubert.permute(0, 2, 1).squeeze(0).cpu().numpy()

            # extract audio MIDI
            pred = midi_model(x.unsqueeze(0))
            p, i = midi_decoder.decode(pred, audio=x.unsqueeze(0))

        # convert audio MIDI to a sequence
        scale_factor = midi_config["hop_length"] / midi_config["sample_rate"]
        time = (np.array(i) * scale_factor).reshape(-1, 2)
        if p is None:
            logging.info(f"skipping {utt_id} since it has no MIDI information")
            continue

        midi = np.array([round(midi) for midi in p])
        T = audio.shape[-1] / config["sampling_rate"]
        midi = midi_to_frames(midi, time, T, shift_ms=config["shiftms"])

        if midi is None:
            logging.info(
                f"skipping {utt_id} since it has no extracted MIDI information"
            )
            continue

        est_lf0_score = _midi_to_hz(midi, log_f0=True)

        est_lf0_score = np.expand_dims(est_lf0_score, axis=-1)
        if not args.skip_gtmidi:
            gt_lf0_score = np.expand_dims(gt_lf0_score, axis=-1)
        else:
            gt_lf0_score = np.expand_dims(est_lf0_score, axis=-1)

        midi = np.expand_dims(midi, axis=-1)

        # Calculate the F0 Fluctuation
        time = t
        f0_normed = f0 / maxf0
        spliner = UnivariateSpline(time, f0_normed, s=10)
        f0_smoothened = spliner(time)
        f0_fluc = []

        for i in range(len(f0_smoothened)):
            fluc = (f0_normed[i] - f0_smoothened[i])
            f0_fluc.append(fluc)

        f0_fluc = np.array(f0_fluc)

        logging.info(f"hubert: {hubert.shape}")
        logging.info(f"logmel: {logmel.shape}")
        logging.info(f"f0: {f0.shape}")
        logging.info(f"f0 fluctuation: {f0_fluc.shape}")
        logging.info(f"loud: {loud.shape}")
        logging.info(f"audio: {audio.shape}")
        logging.info(f"gt score: {gt_lf0_score.shape}")
        logging.info(f"est score: {est_lf0_score.shape}")
        logging.info(f"raw midi: {midi.shape}")

        min_length = min(len(loud), len(midi), len(hubert))
        hubert = hubert[:min_length]
        midi = midi[:min_length]
        gt_lf0_score = gt_lf0_score[:min_length]
        est_lf0_score = est_lf0_score[:min_length]
        loud = loud[:min_length]
        logmel = logmel[:min_length]
        f0 = f0[:min_length]
        f0_fluc = f0_fluc[:min_length]
        vuv = vuv[:min_length]
        logging.info("*" * 50)

        logging.info(f"hubert: {hubert.shape}")
        logging.info(f"logmel: {logmel.shape}")
        logging.info(f"f0: {f0.shape}")
        logging.info(f"f0 fluctuation: {f0_fluc.shape}")
        logging.info(f"vuv: {vuv.shape}")
        logging.info(f"loud: {loud.shape}")
        logging.info(f"audio: {audio.shape}")
        logging.info(f"gt score: {gt_lf0_score.shape}")
        logging.info(f"est score: {est_lf0_score.shape}")
        logging.info(f"raw midi: {midi.shape}")

        # save features
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "wave",
            audio.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "hubert",
            hubert.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "logmel",
            logmel.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "loud",
            loud.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "gt_lf0_score",
            gt_lf0_score.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "est_lf0_score",
            est_lf0_score.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "f0",  # raw F0
            f0.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "f0_fluc",  # raw F0
            f0_fluc.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "vuv",  # raw VUV
            vuv.astype(np.float32),
        )
        write_hdf5(
            os.path.join(args.dumpdir, f"{utt_id}.h5"),
            "midi",  # raw MIDI
            midi.astype(np.float32),
        )


if __name__ == "__main__":
    main()
