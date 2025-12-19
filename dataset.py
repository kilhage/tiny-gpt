"""
Download a starter text dataset for tiny-gpt.

By default this grabs the Tiny Shakespeare corpus (~1 MB) and saves it to
`data.txt` in the repository root. The script is lightweight (uses stdlib
only), retries on transient failures, and writes atomically via a temp file.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
DEFAULT_DEST = Path(__file__).resolve().parent / "data.txt"
DEFAULT_CHUNK_MB = 1


def download_dataset(
    url: str = DEFAULT_URL,
    dest: Path = DEFAULT_DEST,
    *,
    force: bool = False,
    retries: int = 5,
    backoff: float = 2.0,
    chunk_size: int | None = None,
) -> Path:
    """Download a text dataset with simple retries and atomic writes."""
    dest = dest.expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"Skip: {dest} already exists")
        return dest

    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    chunk_bytes = chunk_size or DEFAULT_CHUNK_MB * 1024 * 1024

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "tiny-gpt-dataset/1.0"})
            with urlopen(req, timeout=30) as resp:
                status = getattr(resp, "status", 200)
                if status >= 400:
                    raise HTTPError(url, status, resp.reason, resp.headers, None)

                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_bytes)
                        if not chunk:
                            break
                        f.write(chunk)

            tmp_path.replace(dest)
            print(f"Downloaded dataset to: {dest}")
            return dest

        except (HTTPError, URLError, OSError) as err:
            err_msg = f"Attempt {attempt}/{retries} failed: {err}"
            print(err_msg, file=sys.stderr)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

            if attempt == retries:
                raise

            sleep_for = backoff**attempt
            print(f"Retrying in {sleep_for:.1f}s...", file=sys.stderr)
            time.sleep(sleep_for)

    return dest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a text dataset to data.txt " "(Tiny Shakespeare by default)."
        )
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="Source URL for the dataset (default: Tiny Shakespeare).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DEST,
        help="Where to save the dataset (default: repo data.txt).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the output file already exists.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Maximum retry attempts on failure.",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=2.0,
        help=("Base seconds for exponential backoff " "(sleep = backoff**attempt)."),
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=DEFAULT_CHUNK_MB,
        help="Download chunk size in MB (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk_size = max(1, args.chunk_size_mb) * 1024 * 1024
    download_dataset(
        url=args.url,
        dest=args.output,
        force=args.force,
        retries=args.retries,
        backoff=args.backoff,
        chunk_size=chunk_size,
    )


if __name__ == "__main__":
    main()
