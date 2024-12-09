import argparse
from dataclasses import dataclass
from pathlib import Path

from src.trainer import Metadata, Trainer, TrainerConfig


@dataclass
class Args:
    save_dir: Path
    seed: int
    layer: int
    model_name: str
    lang: list[str]
    data_dir: Path
    hid_feats: int
    sparsity_coef: float
    batch_size: int
    buffer_size: int
    preload_factor: int
    n_epochs: int
    log_path: Path | None

    @staticmethod
    def from_argparse() -> "Args":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "dest",
            type=str,
            help="Directory to save parameters and results",
            metavar="DEST",
        )
        parser.add_argument(
            "-s",
            "--seed",
            default=0,
            type=int,
            help="Random seed. Defaults to 0",
            metavar="UINT",
        )
        parser.add_argument(
            "-l",
            "--layer",
            required=True,
            type=int,
            help="Layer on which SAE is trained",
            metavar="LAYER",
        )
        parser.add_argument(
            "-name",
            "--model-name",
            required=True,
            type=str,
            help="HF name of the model",
            metavar="STRING",
        )
        parser.add_argument(
            "-lang",
            "--lang",
            required=True,
            type=str,
            help="Languages on whose activations SAE is trained. A comma-separated list, e.g. zh,en",
            metavar="LANG,...",
        )
        parser.add_argument(
            "-ds",
            "--data-dir",
            required=True,
            type=str,
            help="Directory that stores activations in subdir named <model>-<lang>",
            metavar="PATH",
        )
        parser.add_argument(
            "-feats",
            "--hid-feats",
            required=True,
            type=int,
            help="Number of hidden features of SAE",
            metavar="UINT",
        )
        parser.add_argument(
            "-lmbda",
            "--sparsity-coef",
            required=True,
            type=float,
            help="Sparsity coefficient",
            metavar="FLOAT",
        )
        parser.add_argument(
            "-bs",
            "--batch-size",
            required=True,
            type=int,
            help="Batch size",
            metavar="UINT",
        )
        parser.add_argument(
            "-n",
            "--buffer-size",
            required=True,
            type=int,
            help="Number of activations to load into GPU",
            metavar="UINT",
        )
        parser.add_argument(
            "-preload",
            "--preload-factor",
            required=True,
            type=int,
            help="Preload UINT x BUF_SIZE into CPU memory",
            metavar="UINT",
        )
        parser.add_argument(
            "-epochs",
            "--n-epochs",
            required=True,
            type=int,
            help="Number of epochs",
            metavar="UINT",
        )
        parser.add_argument(
            "-log",
            "--log",
            required=False,
            type=str,
            help="Path to log file",
            metavar="PATH",
        )
        args = parser.parse_args()

        save_dir = Path(args.dest).resolve()
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        data_dir = Path(args.data_dir).resolve()
        assert data_dir.exists(), f"{data_dir} doesn't exist!"

        if args.log is None:
            log_path = None
        else:
            log_path = Path(args.log).resolve()

        return Args(
            save_dir=save_dir,
            seed=args.seed,
            layer=args.layer,
            model_name=args.model_name,
            lang=args.lang.split(","),
            data_dir=data_dir,
            hid_feats=args.hid_feats,
            sparsity_coef=args.sparsity_coef,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            preload_factor=args.preload_factor,
            n_epochs=args.n_epochs,
            log_path=log_path,
        )


def main(args: Args) -> None:
    config = TrainerConfig(
        metadata=Metadata(model_name=args.model_name, layer=args.layer, lang=args.lang),
        hid_feats=args.hid_feats,
        sparsity_coef=args.sparsity_coef,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
    )
    trainer = Trainer(
        config,
        args.data_dir,
        args.preload_factor,
        args.log_path,
        args.save_dir,
        args.seed,
    )
    trainer.train(args.n_epochs)


if __name__ == "__main__":
    args = Args.from_argparse()
    main(args)
