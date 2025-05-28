# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import logging
import os

from multiprocessing import Manager

import numpy as np

from torch.utils.data import Dataset

from serenade.utils import find_files, read_hdf5, get_basename


class FeatsDataset(Dataset):
    """PyTorch compatible audio and mel dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*.h5",
        scaler=None,
        return_utt_id=False,
        allow_cache=False,
        score_type="est_lf0_score",
        logmel_type="logmel",
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            audio_load_fn (func): Function to load audio file.
            mel_load_fn (func): Function to load feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in audio_files]
        logging.info(f"score type: {score_type}")
        self.utt_ids = utt_ids
        self.audio_files = audio_files
        self.scaler = scaler
        self.score_load_fn = lambda x: read_hdf5(x, score_type)  # NOQA
        self.audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
        self.logmel_load_fn = lambda x: read_hdf5(x, logmel_type)  # NOQA
        self.loud_load_fn = lambda x: read_hdf5(x, "loud")  # NOQA
        self.midi_load_fn = lambda x: read_hdf5(x, "midi")  # NOQA
        self.hubert_load_fn = lambda x: read_hdf5(x, "hubert")  # NOQA
        self.lf0_load_fn = lambda x: read_hdf5(x, "f0")  # NOQA
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])
        hubert = self.hubert_load_fn(self.audio_files[idx])
        logmel = self.logmel_load_fn(self.audio_files[idx])
        score = self.score_load_fn(self.audio_files[idx])
        midi = self.midi_load_fn(self.audio_files[idx])
        loud = self.loud_load_fn(self.audio_files[idx])
        lf0 = self.lf0_load_fn(self.audio_files[idx])

        # normalize the data
        if self.scaler is not None:
            logmel = (logmel - self.scaler["logmel"].mean_) / self.scaler[
                "logmel"
            ].scale_
            hubert = (hubert - self.scaler["hubert"].mean_) / self.scaler[
                "hubert"
            ].scale_
            score = (score - self.scaler["score"].data_min_) / (
                self.scaler["score"].data_max_ - self.scaler["score"].data_min_
            )
            # score = (score - self.scaler["lf0"].data_min_) / (self.scaler["lf0"].data_max_ - self.scaler["lf0"].data_min_)
            loud = (loud - self.scaler["loud"].data_min_) / (
                self.scaler["loud"].data_max_ - self.scaler["loud"].data_min_
            )

            # debugger
            if np.isnan(logmel).any():
                logging.info(f"contains nan: {utt_id}")

        items = {
            "audio": audio,
            "logmel": logmel,
            "hubert": hubert,
            "loud": loud,
            "score": score,
            "midi": midi,
            "lf0": lf0,
        }

        if self.return_utt_id:
            items["utt_id"] = utt_id

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class FeatsDatasetNew(Dataset):
    """PyTorch compatible audio and mel dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*.h5",
        scaler=None,
        return_utt_id=False,
        allow_cache=False,
        score_type="est_lf0_score",
        logmel_type="logmel",
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            audio_load_fn (func): Function to load audio file.
            mel_load_fn (func): Function to load feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in audio_files]
        logging.info(f"score type: {score_type}")
        self.utt_ids = utt_ids
        self.audio_files = audio_files
        self.scaler = scaler
        self.score_load_fn = lambda x: read_hdf5(x, score_type)  # NOQA
        self.audio_load_fn = lambda x: read_hdf5(x, "wave")  # NOQA
        self.logmel_load_fn = lambda x: read_hdf5(x, logmel_type)  # NOQA
        self.loud_load_fn = lambda x: read_hdf5(x, "loud")  # NOQA
        self.midi_load_fn = lambda x: read_hdf5(x, "midi")  # NOQA
        self.hubert_load_fn = lambda x: read_hdf5(x, "hubert")  # NOQA
        self.lf0_load_fn = lambda x: read_hdf5(x, "f0")  # NOQA
        self.f0_fluc_load_fn = lambda x: read_hdf5(x, "f0_fluc")  # NOQA
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])
        hubert = self.hubert_load_fn(self.audio_files[idx])
        logmel = self.logmel_load_fn(self.audio_files[idx])
        score = self.score_load_fn(self.audio_files[idx])
        midi = self.midi_load_fn(self.audio_files[idx])
        loud = self.loud_load_fn(self.audio_files[idx])
        lf0 = self.lf0_load_fn(self.audio_files[idx])
        f0_fluc = self.f0_fluc_load_fn(self.audio_files[idx])

        # normalize the data
        if self.scaler is not None:
            logmel = (logmel - self.scaler["logmel"].mean_) / self.scaler[
                "logmel"
            ].scale_
            hubert = (hubert - self.scaler["hubert"].mean_) / self.scaler[
                "hubert"
            ].scale_
            score = (score - self.scaler["score"].data_min_) / (
                self.scaler["score"].data_max_ - self.scaler["score"].data_min_
            )
            # score = (score - self.scaler["lf0"].data_min_) / (self.scaler["lf0"].data_max_ - self.scaler["lf0"].data_min_)
            loud = (loud - self.scaler["loud"].data_min_) / (
                self.scaler["loud"].data_max_ - self.scaler["loud"].data_min_
            )

            # debugger
            if np.isnan(logmel).any():
                logging.info(f"contains nan: {utt_id}")

        items = {
            "audio": audio,
            "logmel": logmel,
            "hubert": hubert,
            "loud": loud,
            "score": score,
            "midi": midi,
            "lf0": lf0,
            "f0_fluc": f0_fluc,
        }

        if self.return_utt_id:
            items["utt_id"] = utt_id

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)
