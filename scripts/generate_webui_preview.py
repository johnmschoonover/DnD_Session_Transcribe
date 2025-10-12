"""Utility to capture the Web UI landing page markup for documentation previews."""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Final

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from fastapi.testclient import TestClient  # noqa: E402

from dnd_session_transcribe.web import app  # noqa: E402

DEFAULT_OUTPUT: Final[pathlib.Path] = pathlib.Path("docs/previews/webui_root.html")


def render_preview(output_path: pathlib.Path) -> pathlib.Path:
    """Render the Web UI landing page and write it to ``output_path``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with TestClient(app, follow_redirects=True) as client:
        response = client.get("/")
    response.raise_for_status()
    output_path.write_text(response.text, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help="Location for the rendered HTML preview (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = render_preview(args.output.resolve())
    print(f"Wrote Web UI preview to {output}")


if __name__ == "__main__":
    main()
