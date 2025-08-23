# src/hf_utils.py
# src/hf_utils.py
from __future__ import annotations
import os
from typing import Optional, Sequence, List, Tuple, Union

from huggingface_hub import hf_hub_download, list_repo_files

__all__ = ["hf_download", "hf_download_robust"]


def hf_download(
    repo_id: str,
    filename: str,
    repo_type: str = "model",
    token: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """
    Backward-compatible thin wrapper around hf_hub_download.
    Several parts of your app previously imported `hf_download` directly.
    """
    # Silence symlink warnings on some hosts (Streamlit Cloud)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=token,
        force_download=force_download,
    )


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
    If all candidates fail, raise with a helpful message listing what
    actually exists in the repo.

    Returns: local filesystem path to the downloaded file (cached).
    """
    # Silence symlink warnings on some hosts (Streamlit Cloud)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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
