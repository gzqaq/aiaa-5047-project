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
        hf_name: str,
        layers_to_store: list[int],
        model_path: Path | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.logger = setup_logger("hooked", log_path)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.logger.info(f"Use device {device}")

        if model_path is None:
            model_name_or_path = hf_name
        else:
            model_name_or_path = f"{model_path}"

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
        target_num: int,
        save_dir: Path | None = None,
        offset: int = 0,
    ) -> None:
        r"""
        Collect activations on the given text data.

        Args:
          text_data: [str]
          max_batch_size: the maximum number of sequences for model inference
          target_num: number of activations to collect for each layer
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
                inputs = {
                    "input_ids": tokens[: self.ctx_len],
                    "attention_mask": [1] * self.ctx_len,
                }
                batch.append(inputs)
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
                n_collected = self.store_collected(attn_mask, save_dir, i + offset)
                total_collected += n_collected

                tm_elapsed = time.time() - tm_beg
                self.log_batch(n_collected, total_collected, target_num, tm_elapsed)

                if total_collected >= target_num:
                    self.logger.info(f"{target_num} collected, exit")
                    return

                # reset
                batch = []

        if total_collected < target_num:
            self.logger.warning(f"Only collected {total_collected} tokens")

    def store_collected(
        self, attn_mask: torch.Tensor, save_dir: Path | None, text_idx: int
    ) -> int:
        mask = attn_mask.cpu().numpy()
        # Qwen tokenizer seems not to have BOS, and tokenizer.encode doesn't insert BOS and EOS, so
        # there is no need to mask BOS here.
        # mask[..., 0] = False

        n_collected = 0
        for lyr, hook in self.hooks.items():
            activations = hook.storage[-1].cpu().numpy()
            collected = activations[mask]
            n_collected = collected.shape[0]

            if save_dir is None:
                self.stored_activations[lyr].append(collected)
            else:
                with open(
                    save_dir / f"l{lyr}-n{n_collected}-t{text_idx + 1}.npy", "wb"
                ) as fd:
                    np.save(fd, collected)

            hook.storage = []

        return n_collected

    def log_batch(
        self,
        n_collected: int,
        total_collected: int,
        total_to_collect: int,
        tm_elapsed: float,
    ) -> None:
        tm_total_est = total_to_collect / total_collected * tm_elapsed
        tm_wait_est = tm_total_est - tm_elapsed

        self.logger.info(f"Collected {n_collected} activations for each layer")
        self.logger.info(
            f"{total_collected}/{total_to_collect} collected so far. "
            f"ETW: {tm_wait_est}s"
        )

    @property
    def ctx_len(self) -> int:
        return self.model.cfg.n_ctx
