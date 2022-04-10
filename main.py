from pytorch_lightning.utilities.cli import LightningCLI
from text_gan.lightning import LitLanguageGAN, TextDataModule


def main():
    LightningCLI(
        LitLanguageGAN,
        TextDataModule,
    )


if __name__ == '__main__':
    main()
