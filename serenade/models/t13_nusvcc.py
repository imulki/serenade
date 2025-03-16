# -*- coding: utf-8 -*-

# Copyright 2023 Lester Violeta (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""
Flow matching-based acoustic model implementation for voice conversion
References:
"""

from typing import Sequence
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import logging
import random
import math

from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from serenade.utils.masking import make_non_pad_mask
from serenade.models.matcha_components.flow_matching import CFM
from espnet2.tts.gst.style_encoder import StyleEncoder


class NUSVC(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        # model params below
        use_spemb=True,
        denoiser_residual_channels=256,
    ):
        """Initialize Diffusion Module.

        Args:
            input_dim (int): Dimension of the inputs.
            output_dim (int): Dimension of the outputs.
            denoiser_residual_channels (int): Dimension of diffusion model hidden units.
            use_spemb (bool): Whether or not to use speaker embeddings.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.denoiser_residual_channels = denoiser_residual_channels

        # ppg/hubert -> latent
        self.encoder = Conv1dResnet(
            in_dim=771,
            hidden_dim=512,
            num_layers=2,
            out_dim=384,
        )
        # latent -> melspec priors
        self.post_encoder = torch.nn.Conv1d(384, 80, 1)

        self.cfm_decoder = CFM(
            in_channels=160,
            out_channels=80
        )

        # style encoder from espnet
        self.gst = StyleEncoder(
            # params from NU SVCC T13
            gst_tokens=50,
            conv_chans_list=(128, 128, 256, 256, 512, 512),
        )

    def forward(
        self,
        x,
        lengths,
        lf0,
        vuv,
        lft,
        targets=None, # for reconstruction and training
        reference=None, # for conversion and inference
        n_timesteps=10,
        temperature=0.667,
        prompt_mask=None,
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

        cond = torch.cat([x, lf0, vuv, lft], axis=2)
        spk_embs = self.gst(reference)
        mu = self.encoder(cond, lengths)

        y_mask = make_non_pad_mask(lengths).to(mu.device).unsqueeze(1)
        mu = self.post_encoder(mu.permute(0, 2, 1)) * y_mask

        if targets is not None:
            loss, _ = self.cfm_decoder(
                x1=targets.permute(0, 2, 1),
                mask=y_mask,
                mu=mu,
                spks=spk_embs,
                )

            # FIXME: 80 mel dim is hardcoded
            prior_loss = torch.sum(0.5 * ((targets.permute(0, 2, 1) - mu) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * 80)

            return loss, prior_loss
        else:
            decoder_outputs = self.cfm_decoder.inference(
                mu,
                y_mask,
                n_timesteps,
                temperature,
                spk_embs,
                )
            #decoder_outputs = decoder_outputs[:, :, :y_max_length]

            return decoder_outputs.permute(0, 2, 1)
        #return mel_.squeeze(0), lf0_, vuv_, lft_


    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)




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
        if self.use_mdn:
            log_pi, log_sigma, mu = self(x, lengths)
            sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
            return mu, sigma
        else:
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