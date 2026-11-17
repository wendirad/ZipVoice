"""Weights & Biases logging for training.

Scalars reach W&B through ``wandb.init(sync_tensorboard=True)``: every value the
training script writes to TensorBoard is mirrored, so no scalar code changes.
`WandbSamples` adds audio: at each validation it synthesizes a fixed list of
sentences with the current model and logs them, so progress can be heard.

The sample list is a TSV, one sample per line, the same format as the
``--test-list`` of ``infer_zipvoice``::

    {name}\t{prompt_text}\t{prompt_wav}\t{text}[\t{caption}]

``prompt_text`` and ``text`` must be in the form the training tokenizer expects
(e.g. phonemes); the optional ``caption`` is the readable text shown in W&B.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def read_sample_list(path: str) -> List[Tuple[str, str, str, str, str]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) not in (4, 5):
                raise ValueError(
                    f"{path}:{line_no}: expected 4 or 5 tab-separated fields "
                    f"(name, prompt_text, prompt_wav, text[, caption]), got {len(fields)}"
                )
            if len(fields) == 4:
                fields.append(fields[3])
            items.append(tuple(fields))
    return items


class WandbSamples:
    def __init__(
        self,
        sample_list: str,
        tokenizer,
        out_dir: Path,
        device: torch.device,
        num_step: int = 16,
        guidance_scale: float = 1.0,
        t_shift: float = 0.5,
        feat_scale: float = 0.1,
        vocoder_path: str = None,
    ):
        self.items = read_sample_list(sample_list)
        self.tokenizer = tokenizer
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_step = num_step
        self.guidance_scale = guidance_scale
        self.t_shift = t_shift
        self.feat_scale = feat_scale
        self.vocoder_path = vocoder_path
        self._vocoder = None
        self._feature_extractor = None
        self._prompts_logged = False

    def _load(self):
        if self._vocoder is None:
            from zipvoice.bin.infer_zipvoice import get_vocoder
            from zipvoice.utils.feature import VocosFbank

            self._vocoder = get_vocoder(self.vocoder_path).to(self.device).eval()
            self._feature_extractor = VocosFbank()

    def _log_prompts(self, wandb):
        prompts = {}
        for name, prompt_text, prompt_wav, _, _ in self.items:
            try:
                prompts[f"prompts/{name}"] = wandb.Audio(
                    prompt_wav, caption=prompt_text
                )
            except Exception as error:  # a bad prompt must not stop training
                logging.warning(f"wandb: cannot log prompt {name}: {error}")
        if prompts:
            wandb.log(prompts, commit=False)
        self._prompts_logged = True

    def log(self, model, step: int) -> None:
        """Synthesize every sample with ``model`` and log the audio at ``step``."""
        import wandb

        from zipvoice.bin.infer_zipvoice import generate_sentence

        if not self._prompts_logged:
            self._log_prompts(wandb)

        net = model.module if isinstance(model, DDP) else model
        was_training = net.training
        net.eval()
        logs = {"samples/step": step}
        try:
            self._load()
            with torch.inference_mode():
                for name, prompt_text, prompt_wav, text, caption in self.items:
                    path = self.out_dir / f"{name}-step{step}.wav"
                    try:
                        generate_sentence(
                            save_path=str(path),
                            prompt_text=prompt_text,
                            prompt_wav=prompt_wav,
                            text=text,
                            model=net,
                            vocoder=self._vocoder,
                            tokenizer=self.tokenizer,
                            feature_extractor=self._feature_extractor,
                            device=self.device,
                            num_step=self.num_step,
                            guidance_scale=self.guidance_scale,
                            t_shift=self.t_shift,
                            feat_scale=self.feat_scale,
                        )
                        logs[f"samples/{name}"] = wandb.Audio(
                            str(path), caption=f"step {step}: {caption}"
                        )
                    except Exception as error:
                        logging.warning(f"wandb: sample {name} failed: {error}")
        except Exception as error:
            logging.warning(f"wandb: samples skipped at step {step}: {error}")
        finally:
            net.train(was_training)
        if len(logs) > 1:
            wandb.log(logs)
            logging.info(f"wandb: logged {len(logs) - 1} samples at step {step}")
