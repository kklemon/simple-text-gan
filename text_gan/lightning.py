import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from einops import rearrange
from torch.utils.data import DataLoader
from text_gan.data import TextDataset
from text_gan.modules import Generator, Discriminator


class TextDataModule(pl.LightningDataModule):
    def __init__(self,
                 path: str,
                 seq_len: int,
                 batch_size: int = 32):
        super().__init__()

        self.path = path
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.dataset = None

    def prepare_data(self):
        self.dataset = TextDataset(
            path=self.path,
            seq_len=self.seq_len
        )

    @staticmethod
    def decode_batch( batch):
        return [TextDataset.decode(x) for x in batch]

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


class LitLanguageGAN(pl.LightningModule):
    def __init__(self,
                 latent_dim: int = 128,
                 hidden_dim: int = 256,
                 num_blocks: int = 5,
                 kernel_size: int = 5,
                 seq_len: int = 32,
                 g_lr: float = 1e-4,
                 d_lr: float = 1e-4,
                 sample_interval: int = 1000,
                 lambda_gp: float = 10.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.g_lr = g_lr
        self.d_lr = d_lr
        self.sample_interval = sample_interval
        self.lambda_gp = lambda_gp

        self.register_buffer('fixed_z', self.sample_latent(16))

        self.generator = Generator(
            latent_dim=latent_dim,
            seq_len=seq_len,
            dim=hidden_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size
        )

        self.discriminator = Discriminator(
            seq_len=seq_len,
            dim=hidden_dim,
            num_blocks=num_blocks,
            kernel_size=kernel_size
        )

    def sample_latent(self, n):
        return torch.randn(n, self.latent_dim)

    def generate(self, z):
        return self.generator(z)

    def discriminate(self, x):
        return self.discriminator(x)

    def forward(self, z):
        return self.generate(z)

    def configure_optimizers(self):
        return (
            torch.optim.Adam(self.generator.parameters(), self.g_lr, betas=(0.5, 0.9)),
            torch.optim.Adam(self.discriminator.parameters(), self.d_lr, betas=(0.5, 0.9)),
        )

    def gradient_penalty(self, reals, fakes):
        alpha = torch.rand((len(reals), 1, 1)).to(reals.device)
        interpolations = (alpha * reals + (1 - alpha) * fakes).requires_grad_(True)
        interpolation_scores = self.discriminate(interpolations)

        fake_labels = torch.ones_like(interpolation_scores, requires_grad=False)

        grads = torch.autograd.grad(
            outputs=interpolation_scores,
            inputs=interpolations,
            grad_outputs=fake_labels,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = grads.view(grads.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(self, batch, batch_idx, optimizer_idx):
        reals = rearrange(F.one_hot(batch, num_classes=256), 'b s d -> b d s').float()
        z = self.sample_latent(len(batch)).to(self.device)

        if optimizer_idx == 0:
            # Optimize G

            fakes = self.generate(z)
            fake_scores = self.discriminate(fakes)
            loss = -fake_scores.mean()

            self.log('g_loss', loss, prog_bar=True)

            return loss
        else:
            # Optimize D

            fakes = self.generate(z)
            fake_scores = self.discriminate(fakes)
            real_scores = self.discriminate(reals)

            loss = -real_scores.mean() + fake_scores.mean() + self.lambda_gp * self.gradient_penalty(reals, fakes)

            self.log('d_loss', loss, prog_bar=True)

            return loss

    def on_train_batch_end(self, *args):
        if (self.global_step + 1) % self.sample_interval == 0:
            self.log_samples()

    def log_samples(self):
        self.generator.eval()

        samples = self.generate(self.fixed_z).argmax(1)
        samples_str = f'```\n' + '\n'.join(TextDataModule.decode_batch(samples)) + '\n```'

        tensorboard = self.logger.experiment
        tensorboard.add_text('samples', samples_str, global_step=self.global_step)

        self.generator.train()
