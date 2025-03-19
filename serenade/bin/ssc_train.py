#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Train singing style conversion model."""

import argparse
import logging
import os
import sys
import time

from collections import defaultdict, OrderedDict

import matplotlib
import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
import yaml

from prettytable import PrettyTable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from joblib import load
from pathlib import Path

import serenade
import serenade.models
import serenade.trainers
import serenade.collaters

from serenade.datasets import FeatsDataset
from serenade.vocoder import Vocoder

from serenade.utils import read_hdf5
from serenade.utils.types import str_or_none

# set to avoid matplotlib error in CLI environment
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from serenade.schedulers.warmup_lr import WarmupLR

scheduler_classes = dict(warmuplr=WarmupLR)


def count_parameters(model):
    # Detailed table showing all parameters
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)

    # Summary table grouped by main module
    summary_table = PrettyTable(["Main Module", "Parameters"])
    module_params = defaultdict(int)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Remove "module." prefix if present
        if name.startswith("module."):
            main_module = name[7:].split(".")[0]
        else:
            main_module = name.split(".")[0]
        module_params[main_module] += parameter.numel()

    for module, params in module_params.items():
        summary_table.add_row([module, params])

    print("\nSummary by main module:")
    print(summary_table)
    print(f"\nTotal Trainable Params: {total_params}")
    return total_params


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description=("Train VC model (See detail in bin/vc_train.py).")
    )
    parser.add_argument(
        "--train-dumpdir",
        required=True,
        type=str,
        help=("directory including source training data. "),
    )
    parser.add_argument(
        "--dev-dumpdir",
        required=True,
        type=str,
        help=("directory including source development data. "),
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="stats file for target denormalization.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="directory to save checkpoints.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file.",
    )
    parser.add_argument(
        "--init-checkpoint",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to initialize pretrained params. (default="")',
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--rank",
        "--local_rank",
        default=0,
        type=int,
        help="rank for distributed training. no need to explictly specify.",
    )
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="Number of processes for distributed training. No need to explicitly specify.",
    )
    args = parser.parse_args()

    # Initialize distributed training settings
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = args.world_size > 1

    args.rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    logging.basicConfig(level=logging.INFO if args.rank == 0 else logging.WARN)
    logging.info(f"World size: {args.world_size}")
    logging.info(f"Local rank: {args.rank}")

    if args.distributed:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.rank = torch.distributed.get_rank()

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        # device = torch.device("cuda")
        device = torch.device(f"cuda:{args.rank}")
        torch.cuda.set_device(args.rank)
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set random seed
    set_seed(0)

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load main config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    # save config
    config["version"] = serenade.__version__  # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # load stats
    scaler = load(args.stats)

    # load target stats for denormalization
    if args.stats is not None:
        config["trg_stats"] = {
            "mean": scaler["logmel"].mean_,
            "scale": scaler["logmel"].scale_,
        }

    # get dataset
    train_dataset = FeatsDataset(
        root_dir=args.train_dumpdir,
        scaler=scaler,
        score_type=config.get("score_type", "est_lf0_score"),
        logmel_type=config.get("logmel_type", "logmel"),
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    dev_dataset = FeatsDataset(
        root_dir=args.dev_dumpdir,
        scaler=scaler,
        score_type=config.get("score_type", "est_lf0_score"),
        logmel_type=config.get("logmel_type", "logmel"),
        allow_cache=config.get("allow_cache", False),  # keep compatibility
    )
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collater_class = getattr(
        serenade.collaters,
        config.get("collater_type", "NARVCCollater"),
    )
    collater = collater_class()
    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler

        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models
    model_class = getattr(
        serenade.models,
        config.get("model_type", "Serenade"),
    )
    # torch.set_float32_matmul_precision("high")
    model = model_class(**config["model_params"]).to(device)

    # load vocoder
    vocoder_type = config["vocoder"].get("vocoder_type", "")
    vocoder = Vocoder(
        config["vocoder"]["checkpoint"],
        config["vocoder"]["config"],
        config["vocoder"]["stats"],
        device,
        trg_stats=config[
            "trg_stats"
        ],  # this is used to denormalized the converted features,
    )

    # define optimizers and schedulers
    optimizer_class = getattr(
        torch.optim,
        # keep compatibility
        config.get("optimizer_type", "Adam"),
    )
    optimizer = optimizer_class(
        model.parameters(),
        **config["optimizer_params"],
    )
    scheduler_class = getattr(
        torch.optim.lr_scheduler,
        # keep compatibility
        config.get("scheduler_type", "StepLR"),
    )
    scheduler = scheduler_class(
        optimizer=optimizer,
        **config["scheduler_params"],
    )

    if args.distributed:
        # wrap model for distributed training
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.rank],
            output_device=args.rank,
            find_unused_parameters=True,
        )
        logging.info("Model wrapped with DistributedDataParallel.")

    # show settings
    logging.info(model)
    logging.info(count_parameters(model))
    logging.info(optimizer)
    logging.info(scheduler)

    # define trainer
    trainer_class = getattr(
        serenade.trainers,
        config.get("trainer_type", "ARVCTrainer"),
    )
    trainer = trainer_class(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        scaler=scaler,
        model=model,
        vocoder=vocoder,
        criterion=None,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.init_checkpoint) != 0:
        trainer.load_checkpoint(args.init_checkpoint, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.init_checkpoint}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")

    # freeze modules if necessary
    if config.get("freeze-mods", None) is not None:
        assert type(config["freeze-mods"]) is list
        trainer.freeze_modules(config["freeze-mods"])
        logging.info(f"Freeze modules with prefixes {config['freeze-mods']}.")

    # run training loop
    try:
        trainer.run()
    finally:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl")
        )
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
