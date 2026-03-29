from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local model artifact to Hugging Face Hub."
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. username/network-traffic-model")
    parser.add_argument("--token", required=True, help="Hugging Face access token with write permission")
    parser.add_argument(
        "--model-path",
        default="model/model.pkl",
        help="Local model file path (default: model/model.pkl)",
    )
    parser.add_argument(
        "--target-filename",
        default="model.pkl",
        help="Filename inside the HF repository (default: model.pkl)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository if it does not exist",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Repository type on Hugging Face (default: model)",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    args = parse_args()
    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    create_repo(
        repo_id=args.repo_id,
        token=args.token,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    api = HfApi(token=args.token)
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=args.target_filename,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
    )

    print("Upload completed.")
    print(f"Repo: https://huggingface.co/{args.repo_id}")
    print(
        "Direct URL for Streamlit secret MODEL_URL:\n"
        f"https://huggingface.co/{args.repo_id}/resolve/main/{args.target_filename}"
    )


if __name__ == "__main__":
    main()


