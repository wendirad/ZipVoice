"""Push training checkpoints to the Hugging Face Hub every N iterations.

Uploads run in the background (``run_as_future``) so training does not wait;
a new upload waits only if the previous one is still running. Authentication
comes from the environment (``HF_TOKEN`` or ``huggingface-cli login``).
"""

import logging
import re
from pathlib import Path

from huggingface_hub import HfApi


class HubPusher:
    def __init__(
        self,
        repo_id: str,
        exp_dir: Path,
        every_n: int,
        save_every_n: int,
        private: bool = False,
        keep_last: int = 5,
    ):
        if every_n <= 0 or every_n % save_every_n:
            raise ValueError(
                f"--push-every-n ({every_n}) must be a positive multiple of "
                f"--save-every-n ({save_every_n}): only saved checkpoints can be pushed"
            )
        self.repo_id = repo_id
        self.exp_dir = Path(exp_dir)
        self.every_n = every_n
        self.keep_last = keep_last
        self.api = HfApi()
        self.api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        self._future = None
        # config and tokens once, so any pushed checkpoint is usable on its own
        for name in ("model.json", "tokens.txt"):
            path = self.exp_dir / name
            if path.is_file():
                self.api.upload_file(
                    path_or_fileobj=str(path), path_in_repo=name, repo_id=repo_id
                )
        logging.info(f"hub: pushing checkpoints to {repo_id} every {every_n} iters")

    def _wait(self):
        if self._future is not None:
            try:
                self._future.result()
            except Exception as error:  # a failed upload must not stop training
                logging.warning(f"hub: upload failed: {error}")
            self._future = None

    def maybe_push(self, step: int) -> None:
        if step % self.every_n:
            return
        path = self.exp_dir / f"checkpoint-{step}.pt"
        if not path.is_file():
            logging.warning(f"hub: {path} not found, nothing pushed")
            return
        self._wait()
        logging.info(f"hub: uploading {path.name} to {self.repo_id}")
        self._future = self.api.run_as_future(self._push_and_prune, path, step)

    def _push_and_prune(self, path: Path, step: int) -> None:
        """Upload one checkpoint, then keep only the newest ``keep_last`` on the Hub."""
        self.api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"checkpoints/{path.name}",
            repo_id=self.repo_id,
            commit_message=f"checkpoint at iter {step}",
        )
        logging.info(f"hub: pushed {path.name}")
        prune_hub_checkpoints(self.api, self.repo_id, self.keep_last)

    def finish(self) -> None:
        self._wait()


def prune_hub_checkpoints(api: HfApi, repo_id: str, keep_last: int) -> list:
    """Delete all but the newest ``keep_last`` checkpoints/checkpoint-<step>.pt.

    Deleted files still use storage while they are in the history, so the
    history is squashed afterwards (needed under a private storage limit).
    """
    if keep_last <= 0:
        return []
    from huggingface_hub import CommitOperationDelete

    pattern = re.compile(r"checkpoints/checkpoint-(\d+)\.pt")
    found = sorted(
        (int(m.group(1)), f)
        for f in api.list_repo_files(repo_id)
        if (m := pattern.fullmatch(f))
    )
    old = [f for _, f in found[:-keep_last]]
    if old:
        api.create_commit(
            repo_id,
            operations=[CommitOperationDelete(path_in_repo=f) for f in old],
            commit_message=f"keep the last {keep_last} checkpoints",
        )
        api.super_squash_history(repo_id=repo_id)
        logging.info(f"hub: deleted {', '.join(old)} (keeping the last {keep_last})")
    return old
