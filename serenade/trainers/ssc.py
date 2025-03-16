#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import logging
import os
import soundfile as sf
import time
import torch
import numpy as np

from serenade.trainers.base import Trainer

# set to avoid matplotlib error in CLI environment
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from serenade.utils import write_hdf5, read_hdf5


class SSCTrainer(Trainer):
    """Customized trainer module for
    singing style conversion training.
    """

    def load_trained_modules(self, checkpoint_path, init_mods):
        if self.config["distributed"]:
            main_state_dict = self.model.module.state_dict()
        else:
            main_state_dict = self.model.state_dict()

        if os.path.isfile(checkpoint_path):
            model_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

            # first make sure that all modules in `init_mods` are in `checkpoint_path`
            modules = filter_modules(model_state_dict, init_mods)

            # then, actually get the partial state_dict
            partial_state_dict = get_partial_state_dict(model_state_dict, modules)

            if partial_state_dict:
                if transfer_verification(main_state_dict, partial_state_dict, modules):
                    print_new_keys(partial_state_dict, modules, checkpoint_path)
                    main_state_dict.update(partial_state_dict)
        else:
            logging.error(f"Specified model was not found: {checkpoint_path}")
            exit(1)

        if self.config["distributed"]:
            self.model.module.load_state_dict(main_state_dict)
        else:
            self.model.load_state_dict(main_state_dict)

    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        xs = batch["xs"].to(self.device)
        ys = batch["ys"].to(self.device)
        lens = batch["lens"].to(self.device)
        louds = batch["louds"].to(self.device)
        scores = batch["scores"].to(self.device)

        # model forward
        ret = self.model(
            xs,
            lens,
            ys,
            scores,
            louds,
        )

        # flow matching loss
        self.total_train_loss["train/vector_loss"] += ret["cfm_loss"].item()
        gen_loss = ret["cfm_loss"]

        # mel prior loss
        if self.steps > self.config.get("prior_loss_start_steps", 0):
            self.total_train_loss["train/prior_loss"] += ret["prior_loss"].item()
            gen_loss += ret["prior_loss"]

        self.total_train_loss["train/loss"] += gen_loss.item()

        # update model
        self.optimizer.zero_grad()
        gen_loss.backward()

        if self.config["grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["grad_norm"],
            )
        self.optimizer.step()
        self.scheduler.step()

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    @torch.no_grad()
    def _generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""

        # define function for plot prob and att_ws
        def _plot_and_save(
            array, figname, figsize=(6, 4), dpi=150, ref=None, origin="upper"
        ):
            shape = array.shape
            if len(shape) == 1:
                # for eos probability
                plt.figure(figsize=figsize, dpi=dpi)
                plt.plot(array)
                plt.xlabel("Frame")
                plt.ylabel("Probability")
                plt.ylim([0, 1])
            elif len(shape) == 2:
                # for tacotron 2 attention weights, whose shape is (out_length, in_length)
                if ref is None:
                    plt.figure(figsize=figsize, dpi=dpi)
                    plt.imshow(array.T, aspect="auto", origin=origin)
                    plt.xlabel("Input")
                    plt.ylabel("Output")
                else:
                    plt.figure(figsize=(figsize[0] * 2, figsize[1]), dpi=dpi)
                    plt.subplot(1, 2, 1)
                    plt.imshow(array.T, aspect="auto", origin=origin)
                    plt.xlabel("Synthesized")
                    plt.ylabel("Output")
                    plt.subplot(1, 2, 2)
                    plt.imshow(ref.T, aspect="auto", origin=origin)
                    plt.xlabel("GT")
                    plt.ylabel("Output")
            elif len(shape) == 4:
                # for transformer attention weights,
                # whose shape is (#leyers, #heads, out_length, in_length)
                plt.figure(
                    figsize=(figsize[0] * shape[0], figsize[1] * shape[1]), dpi=dpi
                )
                for idx1, xs in enumerate(array):
                    for idx2, x in enumerate(xs, 1):
                        plt.subplot(shape[0], shape[1], idx1 * shape[1] + idx2)
                        plt.imshow(x, aspect="auto")
                        plt.xlabel("Input")
                        plt.ylabel("Output")
            else:
                raise NotImplementedError("Support only from 1D to 4D array.")
            plt.tight_layout()
            if not os.path.exists(os.path.dirname(figname)):
                # NOTE: exist_ok = True is needed for parallel process decoding
                os.makedirs(os.path.dirname(figname), exist_ok=True)
            plt.savefig(figname)
            plt.close()

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # generate
        xs = batch["xs"].to(self.device)
        ys = batch["ys"].to(self.device)
        lens = batch["lens"].to(self.device)
        louds = batch["louds"].to(self.device)
        scores = batch["scores"].to(self.device)

        if self.config["distributed"]:
            model_ = self.model.module
        else:
            model_ = self.model

        for idx, (x, y, lns, loud, score) in enumerate(
            zip(xs, ys, lens, louds, scores)
        ):
            start_time = time.time()

            x = x[:lns]
            y = y[:lns]
            loud = loud[:lns]
            score = score[:lns]

            # reshape for single batch
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            lns = torch.tensor([lns], dtype=torch.long, device=self.device)
            loud = loud.unsqueeze(0)
            score = score.unsqueeze(0)

            if not os.path.exists(os.path.join(dirname, "wav")):
                os.makedirs(os.path.join(dirname, "wav"), exist_ok=True)

            # ground truth
            wave, sr = self.vocoder.decode(y.squeeze(0))
            sf.write(
                os.path.join(dirname, "wav", f"{idx}_gt.wav"),
                wave.cpu().numpy(),
                sr,
                "PCM_16",
            )


            # reconstruction
            outs = model_.inference(
                x,
                lns,
                score,
                loud,
                x,
                lns,
                y,
                score,
                loud,
            )
            
            _plot_and_save(
                outs.cpu().numpy(),
                dirname + f"/outs/{idx}_out.png",
                ref=y.cpu().numpy(),
                origin="lower",
            )

            wave, sr = self.vocoder.decode(outs)

            sf.write(
                os.path.join(dirname, "wav", f"{idx}_gen.wav"),
                wave.cpu().numpy(),
                sr,
                "PCM_16",
            )

            if idx >= self.config["num_save_intermediate_results"]:
                break
