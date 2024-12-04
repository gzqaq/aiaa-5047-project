import argparse
from dataclasses import dataclass
from pathlib import Path

from src.data.culturax import CulturaXData
from src.models.hooked import HookedModel


@dataclass
class Args:
    save_dir: Path
    layers: list[int]
    model_path: Path
    model_name: str
    ds_path: Path
    total_activations: int
    batch_size: int

    @staticmethod
    def from_argparse() -> "Args":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "dest",
            type=str,
            help="Directory to save collected activations",
            metavar="DEST",
        )
        parser.add_argument(
            "layers",
            nargs="+",
            type=int,
            help="Layers to collect activations",
            metavar="LAYER",
        )
        parser.add_argument(
            "-model",
            "--model-path",
            required=True,
            type=str,
            help="Path to model ckpt dir (a dir that contains config.json)",
            metavar="PATH",
        )
        parser.add_argument(
            "-name",
            "--model-name",
            required=True,
            type=str,
            help="HF name for the model. Required by transformer_lens",
            metavar="STRING",
        )
        parser.add_argument(
            "-ds",
            "--ds-path",
            required=True,
            type=str,
            help="Path to parquet file that contains text data",
            metavar="PATH",
        )
        parser.add_argument(
            "-n",
            "--total-activations",
            required=True,
            type=int,
            help="Number of activations to collect",
            metavar="NUM",
        )
        parser.add_argument(
            "-bs",
            "--batch-size",
            required=True,
            type=int,
            help="Batch size for model inference",
            metavar="NUM",
        )
        args = parser.parse_args()

        save_dir = Path(args.dest).resolve()
        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        model_path = Path(args.model_path).resolve()
        assert model_path.exists(), f"{model_path} doesn't exist!"

        ds_path = Path(args.ds_path).resolve()
        assert ds_path.exists(), f"{ds_path} doesn't exist!"

        return Args(
            save_dir=save_dir,
            layers=args.layers,
            model_path=model_path,
            model_name=args.model_name,
            ds_path=ds_path,
            total_activations=args.total_activations,
            batch_size=args.batch_size,
        )


def main(args: Args) -> None:
    ds = CulturaXData(args.ds_path)
    model = HookedModel(
        f"{args.model_path}",
        args.model_name,
        args.layers,
    )
    model.collect_activations(
        ds.texts, args.batch_size, args.total_activations, args.save_dir
    )


if __name__ == "__main__":
    args = Args.from_argparse()
    main(args)
