#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import numpy as np
import torch
import torch.nn.functional as F


class SSCCollater(object):
    """Customized collater for Pytorch DataLoader in singing style conversion training."""

    def __init__(self):
        """Initialize customized collater for PyTorch DataLoader."""

    def __call__(self, batch):
        """Convert into batch tensors."""

        def pad_list(xs, pad_value):
            """Perform padding for the list of tensors.

            Args:
                xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
                pad_value (float): Value for padding.

            Returns:
                Tensor: Padded tensor (B, Tmax, `*`).

            Examples:
                >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
                >>> x
                [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
                >>> pad_list(x, 0)
                tensor([[1., 1., 1., 1.],
                        [1., 1., 0., 0.],
                        [1., 0., 0., 0.]])

            """
            n_batch = len(xs)
            max_len = max(x.size(0) for x in xs)
            pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

            for i in range(n_batch):
                pad[i, : xs[i].size(0)] = xs[i]

            return pad

        xs, ys, louds, scores = [], [], [], []

        # time resolution check
        sorted_batch = sorted(batch, key=lambda x: -x["hubert"].shape[0])
        sorted_batch = [batch for batch in sorted_batch if len(batch["hubert"]) < 3000]

        for b in sorted_batch:
            xs.append(b["hubert"])
            ys.append(b["logmel"])
            louds.append(b["loud"])
            scores.append(b["score"])

        # get list of lengths (must be tensor for DataParallel)
        lens = torch.from_numpy(np.array([x.shape[0] for x in xs])).long()

        # perform padding and conversion to tensor
        xs = pad_list([torch.from_numpy(x).float() for x in xs], 0)
        ys = pad_list([torch.from_numpy(y).float() for y in ys], 0)
        louds = pad_list([torch.from_numpy(loud).float() for loud in louds], 0)
        scores = pad_list([torch.from_numpy(score).float() for score in scores], 0)

        items = {
            "xs": xs,
            "lens": lens,
            "ys": ys,
            "louds": louds,
            "scores": scores,
        }

        return items
