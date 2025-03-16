import torch
import torch.nn.functional as F
import logging

from tqdm import tqdm
from serenade.models.matcha_components.decoder import Decoder


class CFM(torch.nn.Module):
    def __init__(
        self,
        in_channels=80,
        out_channels=80,
        solver="euler",
        sigma_min=1e-4,
        spk_embed_dim=256,
        decoder_channels=(512, 512),
        decoder_attention_head_dim=256,
    ):
        super().__init__()
        self.n_feats = in_channels
        self.spk_embed_dim = spk_embed_dim
        self.solver = solver
        self.sigma_min = sigma_min
        self.conditioning_shape = in_channels + out_channels
        self.out_channels = out_channels

        self.estimator = Decoder(
            in_channels=in_channels,
            out_channels=out_channels,
            spk_embed_dim=spk_embed_dim,
            channels=decoder_channels,
            attention_head_dim=decoder_attention_head_dim,
        )


    def forward(self, x1, mask, mu, spks, mask_l=None):
        return self.compute_loss(x1, mask, mu, spks, mask_l)
    

    @torch.inference_mode()
    def inference(self, mu, mask, n_timesteps=10, temperature=0.667, spks=None):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, encoder_output_dim)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            sample: generated outputs
                shape: (batch_size, n_feats, output_dim)
        """
        z = torch.randn((mu.shape[0], self.out_channels, mu.shape[2])).to(mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)

        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, trg_spks=spks)


    def solve_euler(self, x, t_span, mu, mask, trg_spks):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, n_feats)
            spks (torch.Tensor): style embedding.
                shape: (batch_size, spk_emb_dim)
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        sol = []

        #for step in tqdm(range(1, len(t_span)), desc="sample time step", total=len(t_span) - 1):
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, trg_spks)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]
    

    def compute_loss(self, x1, mask, mu, spks, mask_l=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        denoised = self.estimator(y, mask, mu, t.squeeze(), spks)
        if mask_l is not None:
            denoised = denoised * mask_l
            u = u * mask_l

        loss = F.mse_loss(denoised, u, reduction="sum") 
        if mask_l is not None:
            loss = loss / (torch.sum(mask_l) * u.shape[1])
        else:
            loss = loss / (torch.sum(mask) * u.shape[1])
        return loss, y