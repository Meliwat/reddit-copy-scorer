"""Deploy the Gradio demo to the Hugging Face Space.

Uploads the contents of space/ (app, vendored scorer/features, requirements,
and the per-subreddit model bundles) to the Space repo. Auth comes from the
HF_TOKEN env var so no secret is ever committed or pasted into code.

Usage (GPU box, venv active):
    HF_TOKEN=hf_xxx python scripts/deploy_space.py
    # optional: override the target Space
    HF_TOKEN=hf_xxx python scripts/deploy_space.py --repo-id Meliwat93/reddit-copy-scorer
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO = "Meliwat93/reddit-copy-scorer"
SPACE_DIR = Path(__file__).resolve().parent.parent / "space"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO)
    p.add_argument("--folder", type=Path, default=SPACE_DIR)
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN (a fresh HF *write* token) in the environment.",
              file=sys.stderr)
        return 1
    if not args.folder.is_dir():
        print(f"ERROR: {args.folder} not found.", file=sys.stderr)
        return 1

    n_models = len(list((args.folder / "models").glob("*.joblib")))
    print(f"Uploading {args.folder} ({n_models} model bundles) -> "
          f"Space {args.repo_id} ...")
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="space",
        folder_path=str(args.folder),
        commit_message="Deploy: 36 subreddits + word-level explanations + confidence flags",
    )
    print(f"Done. https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
