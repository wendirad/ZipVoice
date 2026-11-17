"""Load a pretrained checkpoint into a model whose vocabulary is larger.

Fine-tuning on a new language can need tokens the pretrained model never had
(here the ejective mark ``ʼ``). Their ids are appended after the pretrained
ones in tokens.txt, so every pretrained token keeps its id and its learned
embedding row; each new row starts as the mean of the pretrained rows.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn


def load_pretrained_grow_vocab(filename: Path, model: nn.Module) -> None:
    logging.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    if next(iter(state)).startswith("module."):
        state = {k.removeprefix("module."): v for k, v in state.items()}

    old = state["embed.weight"]
    target = model.state_dict()["embed.weight"]
    if old.shape[0] > target.shape[0]:
        raise ValueError(
            f"checkpoint has {old.shape[0]} tokens but the token file only "
            f"{target.shape[0]}: use the pretrained tokens.txt as the base"
        )
    if old.shape[0] < target.shape[0]:
        grown = old.new_empty(target.shape)
        grown[: old.shape[0]] = old
        grown[old.shape[0] :] = old.mean(dim=0, keepdim=True)
        state["embed.weight"] = grown
        logging.info(
            f"Grew the text embedding from {old.shape[0]} to {target.shape[0]} "
            f"tokens; new rows start at the mean pretrained embedding"
        )
    model.load_state_dict(state, strict=True)
