# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Utility functions."""

import os
import numpy as np
import torch
import yaml


def load_vocoder(checkpoint, config=None, stats=None):
    """Load trained model.

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.
        stats (str): Statistics file path.

    Return:
        torch.nn.Module: Model instance.

    """
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error
    from serenade.vocoder.models import HiFiGANGenerator

    # get model and load parameters
    generator_type = config.get("generator_type", "HiFiGANGenerator")
    model_class = HiFiGANGenerator
    # workaround for typo #295
    generator_params = {
        k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
        for k, v in config["generator_params"].items()
    }
    model = model_class(**generator_params)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"]
    )

    # check stats existence
    if stats is None:
        dirname = os.path.dirname(checkpoint)
        if config["format"] == "hdf5":
            ext = "h5"
        else:
            ext = "npy"
        if os.path.exists(os.path.join(dirname, f"stats.{ext}")):
            stats = os.path.join(dirname, f"stats.{ext}")

    # load stats
    if stats is not None and generator_type != "VQVAE":
        model.register_stats(stats)

    return model
