import argparse
from dataclasses import dataclass
from pathlib import Path

from src.data.culturax import CulturaXData
from src.models.hooked import HookedModel


@dataclass
class Args:
    save_dir: Path
    layers: list[int]
    model_name: str
    model_path: Path | None
    ds_path: Path
    total_activations: int
    batch_size: int
    offset: int
    log_path: Path | None

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
            "-name",
            "--model-name",
            required=True,
            type=str,
            help="HF name of the model",
            metavar="STRING",
        )
        parser.add_argument(
            "-model",
            "--model-path",
            required=False,
            type=str,
            help="Path to model ckpt dir (a dir that contains config.json). Optional",
            metavar="PATH",
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
        parser.add_argument(
            "-offset",
            "--offset",
            default=0,
            type=int,
            help="Which index to load text data. Defaults to 0, negative integers are treated as 0",
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

        if args.model_path is None:
            model_path = None
        else:
            model_path = Path(args.model_path).resolve()
            assert model_path.exists(), f"{model_path} doesn't exist!"

        ds_path = Path(args.ds_path).resolve()
        assert ds_path.exists(), f"{ds_path} doesn't exist!"

        if args.log is None:
            log_path = None
        else:
            log_path = Path(args.log).resolve()

        return Args(
            save_dir=save_dir,
            layers=args.layers,
            model_name=args.model_name,
            model_path=model_path,
            ds_path=ds_path,
            total_activations=args.total_activations,
            batch_size=args.batch_size,
            offset=max(args.offset, 0),
            log_path=log_path,
        )


def main(args: Args) -> None:
    ds = CulturaXData(args.ds_path, offset=args.offset)
    model = HookedModel(
        args.model_name,
        args.layers,
        args.model_path,
        args.log_path,
    )
    model.collect_activations(
        ds.texts,
        args.batch_size,
        args.total_activations,
        args.save_dir,
        args.offset,
    )


if __name__ == "__main__":
    args = Args.from_argparse()
    main(args)
