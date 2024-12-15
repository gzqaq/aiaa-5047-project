import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.ckpt import SAECheckpoint
from src.viz.viz import Visualizer


@dataclass
class Args:
    ckpt: Path
    zh_act: Path
    en_act: Path
    chunk_size: int
    log_path: Path | None

    @staticmethod
    def from_argparse() -> "Args":
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "ckpt",
            type=str,
            help="Path to checkpoint saved by flax.serialization",
            metavar="CKPT",
        )
        parser.add_argument(
            "-zh",
            "--zh-act",
            type=str,
            help="Path to saved activations on Chinese texts. Should be a .npy file",
            metavar="PATH",
        )
        parser.add_argument(
            "-en",
            "--en-act",
            type=str,
            help="Path to saved activations on English texts. Should be a .npy file",
            metavar="PATH",
        )
        parser.add_argument(
            "-size",
            "--chunk-size",
            type=int,
            help="Chunk size when counting co-occurrence",
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

        ckpt_path = Path(args.ckpt).resolve()
        assert ckpt_path.exists(), f"{ckpt_path} doesn't exist!"

        zh_path = Path(args.zh_act).resolve()
        assert zh_path.exists(), f"{zh_path} doesn't exist!"

        en_path = Path(args.en_act).resolve()
        assert en_path.exists(), f"{en_path} doesn't exist!"

        if args.log is None:
            log_path = None
        else:
            log_path = Path(args.log).resolve()

        return Args(
            ckpt=ckpt_path,
            zh_act=zh_path,
            en_act=en_path,
            chunk_size=args.chunk_size,
            log_path=log_path,
        )


def main(args: Args) -> None:
    ckpt = SAECheckpoint.from_flax_bin(args.ckpt)
    zh_act = np.load(args.zh_act)
    en_act = np.load(args.en_act)
    activations = np.concatenate([zh_act, en_act], axis=0)
    visualizer = Visualizer(
        ckpt,
        activations,
        args.chunk_size,
        n_components=2,
        n_clusters=2,
        run_cluster=True,
        run_tsne=True,
        log_path=args.log_path,
    )
