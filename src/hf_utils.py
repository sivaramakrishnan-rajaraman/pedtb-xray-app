# src/hf_utils.py
from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from huggingface_hub import hf_hub_download, list_repo_files

def hf_download_robust(
    repo_id: str,
    filename_or_list: Union[str, Sequence[str]],
    *,
    repo_type: str = "model",
    token: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """
    Try to download one of several candidate filenames from a HF repo.
    If all candidates fail, raise with a helpful message listing what's actually in the repo.

    Returns: local cached filesystem path to the downloaded file.
    """
    candidates: List[str] = (
        list(filename_or_list) if isinstance(filename_or_list, (list, tuple)) else [str(filename_or_list)]
    )

    errors: List[Tuple[str, str]] = []
    for fname in candidates:
        try:
            return hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                repo_type=repo_type,
                token=token,
                force_download=force_download,
            )
        except Exception as e:
            errors.append((fname, str(e)))

    # Nothing matched → enumerate files in the repo to help debugging
    try:
        present = list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)
    except Exception:
        present = []

    msg_lines = [
        f"Could not find any of {candidates} in {repo_id}.",
        f"Files present in repo: {present if present else '[could not list files]'}",
        "Errors per candidate:",
    ]
    for fname, err in errors:
        msg_lines.append(f" - {fname}: {err}")
    raise FileNotFoundError("\n".join(msg_lines))

