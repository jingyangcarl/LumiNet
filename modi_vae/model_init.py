import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
try:
    from modules.models import Encoder, Decoder
except:
    from modi_vae.models import Encoder, Decoder


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 embed_dim=4,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ema_decay=None,
                 learn_logvar=False,
                 load_checkpoint=True,
                 lr=1e-4,
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=["ckpt_path", "ignore_keys", "colorize_nlabels"])
        self.image_key = image_key
        self.lr = lr

        self.encoder = Encoder(double_z=True, z_channels=4, resolution=256, in_channels=3,
                               out_ch=3, ch=128, ch_mult=[1,2,4,4], num_res_blocks=2,
                               attn_resolutions=[], dropout=0.0)
        self.decoder = Decoder(double_z=True, z_channels=4, resolution=256, in_channels=3,
                               out_ch=3, ch=128, ch_mult=[1,2,4,4], num_res_blocks=2,
                               attn_resolutions=[], dropout=0.0)

        self.quant_conv = nn.Conv2d(2*4, 2*embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, 4, 1)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        if load_checkpoint:
            state_dict = torch.load('/home/xxing/model/ControlNet/checkpoints/main-epoch=00-step=7000.ckpt', map_location=torch.device("cpu"))["state_dict"]
            new_state_dict = {}
            for s in state_dict:
                if "my_vae" in s:
                    new_state_dict[s.replace("my_vae.", "")] = state_dict[s]
            self.load_state_dict(new_state_dict)
            print("Successfully load new auto-encoder")


        # By default, prepare for decoder-only finetuning
        self.freeze_encoder()

    # ---------- core VAE pieces ----------
    def encode(self, x):
        h, hs = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, hs

    def decode(self, z, hs):
        z = self.post_quant_conv(z)
        dec = self.decoder(z, hs)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior, hs = self.encode(input)
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z, hs)
        return dec, posterior

    # ---------- training for decoder only ----------
    @torch.no_grad()
    def _encode_nograd(self, x):
        """Encode without gradients; used to prevent updates to encoder/quant_conv."""
        posterior, hs = self.encode(x)
        # Detach so gradients don't flow back to encoder/quant_conv
        z = posterior.sample().detach()
        hs = [h.detach() if isinstance(h, torch.Tensor) else h for h in hs]
        return z, hs

    def training_step(self, batch, batch_idx):
        # Expect batch to be a dict with 'image' or a tensor directly
        x = batch[self.image_key] if isinstance(batch, dict) else batch  # [B,3,H,W] in [-1,1] or [0,1]
        z, _ = self._encode_nograd(x[:,:3,...])
        _, hs = self._encode_nograd(x[:,3:,...])
        x_hat = self.decode(z, hs)

        # Simple reconstruction loss (L1). If inputs are in [-1,1], it's fine for L1 too.
        rec_loss = F.l1_loss(x_hat, x[:,:3,...])

        # (Optional) small MSE term to stabilize
        mse_loss = F.mse_loss(x_hat, x[:,:3,...])
        loss = rec_loss + 0.1 * mse_loss

        self.log_dict({
            "train/l1": rec_loss,
            "train/mse": mse_loss,
            "train/loss": loss
        }, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.image_key] if isinstance(batch, dict) else batch
        z,_ = self._encode_nograd(x[:,:3,...])
        _, hs = self._encode_nograd(x[:,3:,...])
        
        x_hat = self.decode(z, hs)
        rec_loss = F.l1_loss(x_hat, x[:,:3,...])
        mse_loss = F.mse_loss(x_hat, x[:,:3,...])
        loss = rec_loss + 0.1 * mse_loss
        self.log_dict({
            "val/l1": rec_loss,
            "val/mse": mse_loss,
            "val/loss": loss
        }, prog_bar=True, on_epoch=True, batch_size=x.shape[0])

    def configure_optimizers(self):
        # Only optimize decoder + post_quant_conv
        params = list(self.decoder.parameters()) + list(self.post_quant_conv.parameters())
        opt = torch.optim.Adam(params, lr=self.lr, betas=(0.9, 0.999))
        return opt

    def freeze_encoder(self):
        """Freeze encoder and quant_conv (no grads)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.quant_conv.parameters():
            p.requires_grad = False
        # Ensure decoder & post_quant_conv are trainable
        for p in self.decoder.parameters():
            p.requires_grad = True
        for p in self.post_quant_conv.parameters():
            p.requires_grad = True


# import torch
# import torchvision.transforms as T

# # Compose transforms
# transform = T.Compose([
#     T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # illumination jitter
#     T.Lambda(lambda x: x + 0.05 * torch.randn_like(x))  # Gaussian noise
# ])

# # Example: I is [C,H,W] in [0,1] or [-1,1]
# I_aug = transform(I).clamp(-1, 1)  # keep in valid range
