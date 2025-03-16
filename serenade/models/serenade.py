# -*- coding: utf-8 -*-

# Copyright 2025 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)


"""
Flow-matching-based singing style conversion model implementation
References:
    - https://github.com/MoonInTheRiver/DiffSinger
    - https://github.com/nnsvs/nnsvs
    - https://github.com/shivammehta25/Matcha-TTS
"""

from typing import Sequence
from collections import OrderedDict

import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from joblib import load
import logging
import random

from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from serenade.utils.masking import make_non_pad_mask
from serenade.models.matcha_components.flow_matching import CFM
from serenade.modules.gst.style_encoder import StyleEncoder


class Serenade(torch.nn.Module):
    def __init__(
        self,
        # model params below
        input_dim=768, # cvec
        output_dim=80, # logmel
        encoder_channels=80,
        decoder_channels=512,
        gst_embed_dim=256,
        decoder_attention_head_dim=512,
        mask_size=[0.1, 0.5],
        cfg_prob=0.1,
    ):
        """Initialize Diffusion Module.

        Args:
            input_dim (int): Dimension of the inputs.
            output_dim (int): Dimension of the outputs.
            use_spemb (bool): Whether or not to use speaker embeddings.
            resample_ratio (float): Ratio to align the input and output features.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cfg_prob = cfg_prob

        self.encoder = Conv1dResnet(
            in_dim=input_dim,
            hidden_dim=512,
            num_layers=2,
            out_dim=encoder_channels,
        )

        # speaker embedding projection
        self.gst = StyleEncoder(
            # params from NU SVCC T13
            gst_tokens=50,
            conv_chans_list=(128, 128, 256, 256, 512, 512),
            gst_token_dim=gst_embed_dim,
        )

        # encoder outputs + targets + midi + loudness
        conditioning_dim = output_dim + encoder_channels + 1 + 1

        # flowmatching model
        self.cfm_decoder = CFM(
            in_channels=conditioning_dim + output_dim, # combines the target and input dimensions
            out_channels=output_dim,
            spk_embed_dim=gst_embed_dim,
            decoder_channels=(decoder_channels, decoder_channels),
            decoder_attention_head_dim=decoder_attention_head_dim,
        )
        self.mask_size = mask_size


    def forward(
        self,
        x,
        lengths,
        logmel,
        midi,
        lft,
    ):
        """Calculate forward propagation.

        Args:
            x (Tensor): Batch of padded input conditioning features (B, Lmax, input_dim).
            lengths (LongTensor): Batch of lengths of each input batch (B,).
            targets (Tensor): Batch of padded target features (B, Lmax, output_dim).
            spk (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).

        Returns (training):
            Tensor: Ground truth noise.
            Tensor: Predicted noise.
            LongTensor: Resampled lengths based on upstream feature.
        Returns (inference):
            Tensor: Predicted mel spectrogram.
        """
        ret = {}

        enc_outs = self.encoder(x, lengths)
        ret["gauss_mel"] = enc_outs
        speaker_features = self.gst(logmel)
        mask = make_non_pad_mask(lengths).to(x.device).unsqueeze(1)

        # dynamic masked segment prediction
        mask_size = random.uniform(self.mask_size[0], self.mask_size[1]) * enc_outs.size(1)
        mask_size = int(mask_size)

        seg_start = random.randint(0, enc_outs.size(1) - mask_size)
        seg_end = seg_start + mask_size

        # mask for loss function
        mask_l = mask.clone()
        mask_l[:, :, 0:seg_start] = 0
        mask_l[:, :, seg_end:] = 0

        # mask for conditioning
        mask_c = mask.clone()
        mask_c[:, :, seg_start:seg_end] = 0

        # compute prior loss (no masking) for content features
        prior_loss = torch.sum(0.5 * ((logmel.permute(0, 2, 1) - enc_outs.permute(0, 2, 1)) ** 2 + math.log(2 * math.pi)) * mask)
        ret["prior_loss"] = prior_loss / (torch.sum(mask) * self.output_dim)

        # mask target features
        cond = logmel
        targets = cond * mask_l.permute(0, 2, 1)

        # mask conditioning features
        cond = cond * mask_c.permute(0, 2, 1)

        enc_outs = torch.cat([enc_outs, midi, lft, cond], dim=-1)

        ret["cfm_loss"], _ = self.cfm_decoder(
            x1=targets.permute(0, 2, 1),
            mask=mask,
            mu=enc_outs.permute(0, 2, 1),
            spks=speaker_features,
            mask_l=mask_l,
        )

        return ret


    def inference(
        self,
        x,
        lengths,
        midi,
        lft,
        ref_x,
        ref_lengths,
        ref_logmel,
        ref_midi,
        ref_lft,
    ):
        # inference code
        enc_outs_ = self.encoder(x, lengths)

        ref_speaker_features = self.gst(ref_logmel)

        # specify conditioning features (more flexible for changes in the future)
        cond = ref_logmel

        # add reference features to content features
        ref_enc_outs = self.encoder(ref_x, ref_lengths)
        ref_enc_outs = torch.cat([ref_enc_outs, ref_midi, ref_lft, cond], dim=-1)

        # add zero vector replacing conditioning features, emulating masking
        zero_cond = torch.zeros(enc_outs_.shape[0], enc_outs_.shape[1], cond.shape[-1], device=enc_outs_.device)
        enc_outs = torch.cat([enc_outs_, midi, lft, zero_cond], dim=-1)

        # concatenate reference and source features in time dimension
        enc_outs = torch.cat([ref_enc_outs, enc_outs], dim=1)

        # make mask for decoder (just 1s since its a single segment)
        total_lengths = lengths + ref_lengths
        mask = make_non_pad_mask(total_lengths).to(x.device).unsqueeze(1)

        # run decoder
        # FIXME: hard to do a batch calculation because the reference lengths
        # can be different for each sample
        mel_pred = self.cfm_decoder.inference(
            enc_outs.permute(0, 2, 1),
            mask,
            spks=ref_speaker_features,
        ).permute(0, 2, 1)

        # discard reference part
        # FIXME: not very efficient when batched
        mel_pred = mel_pred[:, ref_lengths[0]:, :]

        return mel_pred.squeeze(0)


class Conv1dResnet(torch.nn.Module):
    """Conv1d + Resnet

    The model is inspired by the MelGAN's model architecture (:cite:t:`kumar2019melgan`).
    MDN layer is added if use_mdn is True.

    Args:
        in_dim (int): the dimension of the input
        hidden_dim (int): the dimension of the hidden state
        out_dim (int): the dimension of the output
        num_layers (int): the number of layers
        init_type (str): the type of weight initialization
        use_mdn (bool): whether to use MDN or not
        num_gaussians (int): the number of gaussians in MDN
        dim_wise (bool): whether to use dim-wise or not
        in_ph_start_idx (int): the start index of phoneme identity in a hed file
        in_ph_end_idx (int): the end index of phoneme identity in a hed file
        embed_dim (int): the dimension of the phoneme embedding
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layers=4,
        init_type="none",
        use_mdn=False,
        num_gaussians=8,
        dim_wise=False,
        in_ph_start_idx: int = 1,
        in_ph_end_idx: int = 50,
        embed_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_mdn = use_mdn
        self.in_ph_start_idx = in_ph_start_idx
        self.in_ph_end_idx = in_ph_end_idx
        self.num_vocab = in_ph_end_idx - in_ph_start_idx
        self.embed_dim = embed_dim

        if "dropout" in kwargs:
            warn(
                "dropout argument in Conv1dResnet is deprecated"
                " and will be removed in future versions"
            )

        if self.embed_dim is not None:
            assert in_dim > self.num_vocab
            self.emb = nn.Embedding(self.num_vocab, embed_dim)
            self.fc_in = nn.Linear(in_dim - self.num_vocab, embed_dim)
            conv_in_dim = embed_dim
        else:
            conv_in_dim = in_dim

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(conv_in_dim, hidden_dim, kernel_size=7, padding=0),
        ]
        for n in range(num_layers):
            model.append(ResnetBlock(hidden_dim, dilation=2 ** n))

        last_conv_out_dim = hidden_dim if use_mdn else out_dim
        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(hidden_dim, last_conv_out_dim, kernel_size=7, padding=0),
        ]

        self.model = nn.Sequential(*model)

        if self.use_mdn:
            self.mdn_layer = MDNLayer(
                in_dim=hidden_dim,
                out_dim=out_dim,
                num_gaussians=num_gaussians,
                dim_wise=dim_wise,
            )
        else:
            self.mdn_layer = None

        init_weights(self, init_type)


    def forward(self, x, lengths=None, y=None):
        """Forward step

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor
            y (torch.Tensor): the optional target tensor

        Returns:
            torch.Tensor: the output tensor
        """

        if self.embed_dim is not None:
            x_first, x_ph_onehot, x_last = torch.split(
                x,
                [
                    self.in_ph_start_idx,
                    self.num_vocab,
                    self.in_dim - self.num_vocab - self.in_ph_start_idx,
                ],
                dim=-1,
            )
            x_ph = torch.argmax(x_ph_onehot, dim=-1)
            # Make sure to have one-hot vector
            assert (x_ph_onehot.sum(-1) <= 1).all()
            x = self.emb(x_ph) + self.fc_in(torch.cat([x_first, x_last], dim=-1))

        out = self.model(x.transpose(1, 2)).transpose(1, 2)

        if self.use_mdn:
            return self.mdn_layer(out)
        else:
            return out

    def inference(self, x, lengths=None):
        """Inference step

        Find the most likely mean and variance if use_mdn is True

        Args:
            x (torch.Tensor): the input tensor
            lengths (torch.Tensor): the lengths of the input tensor

        Returns:
            tuple: mean and variance of the output features
        """
        return self(x, lengths)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class ResnetBlock(torch.nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.LeakyReLU(0.2),
            torch.nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            torch.nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


# Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Args:
        net (torch.nn.Module): network to initialize
        init_type (str): the name of an initialization method:
            normal | xavier | kaiming | orthogonal | none.
        init_gain (float): scaling factor for normal, xavier and orthogonal.
    """
    if init_type == "none":
        return

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)