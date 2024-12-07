import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logging import Logger, setup_logger


class StoreActivationHook:
    storage: list[torch.Tensor]

    def __init__(self) -> None:
        self.storage = []

    def hook_fn(self, module: nn.Module, inp: Any, outp: torch.Tensor) -> None:
        self.storage.append(outp)


class HookedModel:
    model: HookedTransformer
    stored_activations: dict[int, list[np.ndarray]]
    logger: Logger

    def __init__(
        self,
        model_name_or_path: str,
        hf_name: str,
        layers_to_store: list[int],
        log_path: Path | None = None,
    ) -> None:
        self.logger = setup_logger("hooked", log_path)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.logger.info(f"Use device {device}")

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = HookedTransformer.from_pretrained(
            hf_name,
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=device,
            first_n_layers=max(layers_to_store) + 1,
        )
        self.logger.debug("Hooked model initialized")

        self.hooks: dict[int, StoreActivationHook]
        self._init_hooks_and_storage(layers_to_store)

    def _init_hooks_and_storage(self, layers_to_store: list[int]) -> None:
        self.hooks = {}
        self.stored_activations = {}

        for lyr in layers_to_store:
            self.hooks[lyr] = StoreActivationHook()
            self.stored_activations[lyr] = []
            self.model.blocks[lyr].register_forward_hook(self.hooks[lyr].hook_fn)
            self.logger.debug(f"Hook on layer {lyr} registered")

    def collect_activations(
        self,
        text_data: Iterable[str],
        max_batch_size: int,
        total_activations_to_collect: int,
        save_dir: Path | None = None,
        offset: int = 0,
    ) -> None:
        r"""
        Collect activations on the given text data.

        Args:
          text_data: [str]
          max_batch_size: the maximum number of sequences for model inference
          total_activations_to_collect: number of activations to collect for each layer
          save_dir: directory to save collected activations, optional. If not given, all
            collected activations are stored in `self.stored_activations`. If given, make
            sure this directory exists, and activations are saved per batch.
        """
        batch = []
        total_collected = 0
        tm_beg = time.time()
        for i, text in enumerate(text_data):
            tokens = self.model.tokenizer.encode(text)

            # skip texts that are too long to fit in a batch
            if len(tokens) >= self.ctx_len * max_batch_size:
                continue

            while len(tokens) > self.ctx_len:
                batch.append(
                    {
                        "input_ids": tokens[: self.ctx_len],
                        "attention_mask": [1] * self.ctx_len,
                    }
                )
                tokens = tokens[self.ctx_len :]

            if len(tokens) > 0:
                batch.append({"input_ids": tokens, "attention_mask": [1] * len(tokens)})

            if len(batch) > max_batch_size:
                self.logger.info("Reach max_batch_size, start collecting...")

                # collect
                inputs = self.model.tokenizer.pad(batch, return_tensors="pt")
                input_ids = inputs["input_ids"]
                attn_mask = inputs["attention_mask"] == 1

                with torch.inference_mode():
                    self.model(input_ids)

                # store
                n_collected = 0
                for lyr in self.hooks.keys():
                    collected = self.hooks[lyr].storage[-1][attn_mask].cpu().numpy()
                    n_collected = collected.shape[0]

                    if save_dir is None:
                        self.stored_activations[lyr].append(collected)
                    else:
                        with open(
                            save_dir / f"lyr{lyr}-n{n_collected}-t{i + offset + 1}.npy",
                            "wb",
                        ) as fd:
                            np.save(fd, collected)

                    self.hooks[lyr].storage = []

                total_collected += n_collected
                tm_elapsed = time.time() - tm_beg
                tm_total_est = (
                    total_activations_to_collect / total_collected * tm_elapsed
                )
                tm_wait_est = tm_total_est - tm_elapsed
                self.logger.info(
                    f"Collected {n_collected} activations for each layer in this batch"
                )
                self.logger.info(
                    f"{total_collected}/{total_activations_to_collect} collected so far. "
                    f"ETW: {tm_wait_est}s"
                )

                if total_collected >= total_activations_to_collect:
                    self.logger.info(f"{total_activations_to_collect} collected, exit")
                    return

                # reset
                batch = []

        if total_collected < total_activations_to_collect:
            self.logger.warning(f"Only collected {total_collected} tokens")

    @property
    def ctx_len(self) -> int:
        return self.model.cfg.n_ctx
