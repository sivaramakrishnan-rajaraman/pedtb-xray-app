from __future__ import annotations
import os
from typing import Optional
from huggingface_hub import hf_hub_download

def hf_download(
    repo_id: str,
    filename: str,
    repo_type: str = "model",
    token: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """
    Download a file from Hugging Face Hub and return the local file path.
    Works for public (token=None) or private (token required) repos.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        token=token,
        force_download=force_download,
    )
    return local_path
