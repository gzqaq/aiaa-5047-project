import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.ckpt import SAECheckpoint
from src.utils.logging import setup_logger
from src.viz.viz import Visualizer

_LOGGER = setup_logger("script")
_COLOR_MAP = {"zh": "#ff7f0e", "en": "#1f77b4"}


@dataclass
class Args:
    ckpt: Path
    seed: int
    zh_act: Path
    en_act: Path
    chunk_size: int
    n_chunks: int
    affinity_measure: str
    log_path: Path | None
    figsize: tuple[float, float] = (6.4, 4.8)
    dpi: float = 100.0
    alpha: float = 1.0
    marker_size: float = 0.3

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
            "-seed", "--seed", default=42, type=int, help="Random seed", metavar="UINT"
        )
        parser.add_argument(
            "-zh",
            "--zh-act",
            required=True,
            type=str,
            help="Path to saved activations on Chinese texts. Should be a .npy file",
            metavar="PATH",
        )
        parser.add_argument(
            "-en",
            "--en-act",
            required=True,
            type=str,
            help="Path to saved activations on English texts. Should be a .npy file",
            metavar="PATH",
        )
        parser.add_argument(
            "-size",
            "--chunk-size",
            required=True,
            type=int,
            help="Chunk size when counting co-occurrence",
            metavar="UINT",
        )
        parser.add_argument(
            "-chunks",
            "--n-chunks",
            default=0,
            type=int,
            help="Number of chunks per language to run. 0 means as many as possible",
            metavar="UINT",
        )
        parser.add_argument(
            "-affinity",
            "--affinity-measure",
            default="phi-coef",
            type=str,
            help="Which affinity measure to use for clustering. Defaults to phi-coef",
            metavar="MEASURE",
        )
        parser.add_argument(
            "-log",
            "--log",
            required=False,
            type=str,
            help="Path to log file",
            metavar="PATH",
        )
        parser.add_argument(
            "-figsize",
            "--figsize",
            default="6.4,4.8",
            type=str,
            help="Width, height in inches. Separated by comma, defaults to 6.4,4.8",
            metavar="W,H",
        )
        parser.add_argument(
            "-dpi",
            "--dpi",
            default=100.0,
            type=float,
            help="Resolution of the figure in dots-per-inch. Defaults to 100.0",
            metavar="DPI",
        )
        parser.add_argument(
            "-alpha",
            "--alpha",
            default=1.0,
            type=float,
            help="The alpha blending value, between 0 and 1. Defaults to 1.0",
            metavar="ALPHA",
        )
        parser.add_argument(
            "-marker",
            "--marker-size",
            default=0.3,
            type=float,
            help="The marker size in points^2. Defaults to 0.3",
            metavar="SIZE",
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

        w, h = tuple(map(float, args.figsize.split(",")))

        return Args(
            ckpt=ckpt_path,
            seed=args.seed,
            zh_act=zh_path,
            en_act=en_path,
            chunk_size=args.chunk_size,
            n_chunks=args.n_chunks,
            affinity_measure=args.affinity_measure,
            log_path=log_path,
            figsize=(w, h),
            dpi=args.dpi,
            alpha=args.alpha,
            marker_size=args.marker_size,
        )


def main(args: Args) -> None:
    np.random.seed(args.seed)

    ckpt = SAECheckpoint.from_flax_bin(args.ckpt)
    _LOGGER.info(f"SAE weights loaded from {args.ckpt}")
    zh_act = np.load(args.zh_act)
    _LOGGER.info(f"Activations on Chinese texts loaded from {args.zh_act}")
    en_act = np.load(args.en_act)
    _LOGGER.info(f"Activations on English texts loaded from {args.en_act}")

    if args.n_chunks > 0:
        n_chunks = args.n_chunks
    else:
        n_chunks = len(zh_act) // args.chunk_size * args.chunk_size

    zh_act = zh_act[: n_chunks * args.chunk_size]
    en_act = en_act[: n_chunks * args.chunk_size]
    activations = np.concatenate([zh_act, en_act], axis=0)
    _LOGGER.info(f"Will use {len(activations)} activations")

    visualizer = Visualizer(
        ckpt,
        activations,
        args.chunk_size,
        n_components=2,
        n_clusters=2,
        run_cluster=False,
        run_tsne=False,
        log_path=args.log_path,
    )

    # visualize 2d
    visualizer.run_tsne()
    assert visualizer.valid_feats_2d is not None

    # which label zh or en corresponds to
    sae_acts = np.array(ckpt.sae_fwd(activations)[1])
    sae_acts_on_valid = sae_acts[:, visualizer.valid_feats_mask]
    acts_on_valid = {
        "zh": sae_acts_on_valid[: len(zh_act)],
        "en": sae_acts_on_valid[len(zh_act) :],
    }

    labels = run_cluster_get_labels(
        visualizer,
        args.affinity_measure,
        acts_on_valid,
        n_chunks,
        args.chunk_size,
    )

    fig, ax = plt.subplots(1, 1, figsize=args.figsize, dpi=args.dpi)
    for lbl in range(visualizer.cluster_alg.n_clusters):
        label = labels[lbl]
        mask = visualizer.masks[lbl]
        feats = visualizer.valid_feats_2d[mask]
        ax.scatter(
            feats[:, 0],
            feats[:, 1],
            label=label,
            alpha=args.alpha,
            s=args.marker_size,
            c=_COLOR_MAP[label],
        )
    ax.legend()

    fig_path = args.ckpt.with_suffix(f".{args.affinity_measure}.2d.pdf")
    fig.savefig(fig_path)
    _LOGGER.info(f"Visualized in {fig_path}")


def run_cluster_get_labels(
    visualizer: Visualizer,
    affinity_measure: str,
    acts_on_valid: dict[str, np.ndarray],
    n_chunks: int,
    chunk_size: int,
) -> list[str]:
    """
    Run VISUALIZER.run_cluster with AFFINITY_MEASURE, and detect the order of zh and en as
    labels using ACTS_ON_VALID whose keys are zh and en, N_CHUNKS, and CHUNK_SIZE. Returns [zh, en]
    or [en, zh].
    """
    visualizer.run_cluster(affinity_measure)

    acts_on_lbl: dict[str, dict[int, np.ndarray]] = {}
    for lang in ["zh", "en"]:
        acts_on_lbl[lang] = {}
        for lbl in [0, 1]:
            acts_on_lbl[lang][lbl] = acts_on_valid[lang][:, visualizer.masks[lbl]]

    lbl_0_activates_more: dict[str, np.ndarray] = {}
    for lang in ["zh", "en"]:
        lbl_0_activates_more[lang] = np.mean(
            np.greater(
                acts_on_lbl[lang][0]
                .reshape(n_chunks, chunk_size, -1)
                .mean(axis=(-2, -1)),
                acts_on_lbl[lang][1]
                .reshape(n_chunks, chunk_size, -1)
                .mean(axis=(-2, -1)),
            )
        )
        _LOGGER.debug(
            "Proportion of chunks where label-0 features fire more "
            f"on {lang} texts: {lbl_0_activates_more[lang]}"
        )

    if lbl_0_activates_more["zh"] > lbl_0_activates_more["en"]:
        _LOGGER.info("Label 0 corresponds to features firing on Chinese texts")
        labels = ["zh", "en"]
    else:
        _LOGGER.info("Label 1 corresponds to features firing on Chinese texts")
        labels = ["en", "zh"]

    return labels


if __name__ == "__main__":
    main(Args.from_argparse())
