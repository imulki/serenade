#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Calculate statistics of feature files."""

import argparse
import logging
import os

import numpy as np
import yaml

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from joblib import dump

from serenade.datasets import FeatsDataset
from serenade.utils import read_hdf5
from serenade.utils import write_hdf5


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute mean and variance of dumped raw features "
            "(See detail in bin/compute_statistics.py)."
        )
    )
    parser.add_argument(
        "--feats-scp",
        "--scp",
        default=None,
        type=str,
        help=(
            "kaldi-style feats.scp file. "
            "you need to specify either feats-scp or rootdir."
        ),
    )
    parser.add_argument(
        "--rootdir",
        type=str,
        help=(
            "directory including feature files. "
            "you need to specify either feats-scp or rootdir."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--feat_type",
        type=str,
        default="mel",
        help=("feature type. this is used as key name to read h5 feature files. "),
    )
    parser.add_argument(
        "--dumpdir",
        default=None,
        type=str,
        required=True,
        help=(
            "directory to save statistics. if not provided, "
            "stats will be saved in the above root directory. (default=None)"
        ),
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

    # check arguments
    if (args.feats_scp is not None and args.rootdir is not None) or (
        args.feats_scp is None and args.rootdir is None
    ):
        raise ValueError("Please specify either --rootdir or --feats-scp.")

    # check directory existence
    if not os.path.exists(args.dumpdir):
        os.makedirs(args.dumpdir)

    # get dataset
    dataset = FeatsDataset(
        args.rootdir,
        return_utt_id=True,
    )
    logging.info(f"The number of files = {len(dataset)}.")

    # calculate statistics
    scaler = {}
    scaler["hubert"] = StandardScaler()
    scaler["logmel"] = StandardScaler()
    scaler["score"] = MinMaxScaler()
    scaler["loud"] = MinMaxScaler()

    # FIXME: make this more flexible for new features
    for items in tqdm(dataset):
        utt_id = items["utt_id"]
        logmel = items["logmel"]
        hubert = items["hubert"]
        score = items["score"]
        loud = items["loud"]

        logging.info(f"processing {utt_id}")

        scaler["hubert"].partial_fit(hubert)
        scaler["logmel"].partial_fit(logmel)
        scaler["score"].partial_fit(score)
        scaler["loud"].partial_fit(loud)

    # save scaler file
    dump(scaler, os.path.join(args.dumpdir, f"stats.joblib"))
    logging.info(f"Successfully saved statistics file.")


if __name__ == "__main__":
    main()
