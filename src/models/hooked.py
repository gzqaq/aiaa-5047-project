from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


class StoreActivationHook:
    storage: list[torch.Tensor]

    def __init__(self) -> None:
        self.storage = []

    def hook_fn(self, module: nn.Module, inp: Any, outp: torch.Tensor) -> None:
        self.storage.append(outp)


class HookedModel:
    model: HookedTransformer
    stored_activations: dict[int, list[np.ndarray]]

    def __init__(
        self, model_name_or_path: str, hf_name: str, layers_to_store: list[int]
    ) -> None:
        original_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype="auto", device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = HookedTransformer.from_pretrained(
            hf_name, hf_model=original_model, tokenizer=tokenizer
        )

        self.hooks: dict[int, StoreActivationHook]
        self._init_hooks_and_storage(layers_to_store)

    def _init_hooks_and_storage(self, layers_to_store: list[int]) -> None:
        self.hooks = {}
        self.stored_activations = {}

        for lyr in layers_to_store:
            self.hooks[lyr] = StoreActivationHook()
            self.stored_activations[lyr] = []

    def collect_activations(
        self, text_data: Iterable[str], max_batch_size: int
    ) -> None:
        batch = []
        for text in text_data:
            tokens = self.model.tokenizer.encode(text)

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
                # collect
                inputs = self.model.tokenizer.pad(batch, return_tensors="pt")
                input_ids = inputs["input_ids"]
                attn_mask = inputs["attention_mask"] == 1

                with torch.inference_mode():
                    self.model(input_ids)

                # store
                for lyr in self.hooks.keys():
                    self.stored_activations[lyr].append(
                        self.hooks[lyr].storage[-1][attn_mask].cpu().numpy()
                    )
                    self.hooks[lyr].storage = []

                # reset
                batch = []

    @property
    def ctx_len(self) -> int:
        return self.model.cfg.n_ctx
