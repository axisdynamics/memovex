#!/usr/bin/env python3
"""
Install the MemoVex Docker stack into a permanent directory.

After running this script you can safely delete the cloned repository —
the Docker stack lives in its own directory and has no dependency on it.

Usage:
    python3 scripts/setup_docker.py                     # default: ~/memovex
    python3 scripts/setup_docker.py --dir /opt/memovex  # custom path
    python3 scripts/setup_docker.py --start             # install + docker compose up -d
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # repo root
DOCKER_DIR = HERE / "docker"

ENV_TEMPLATE = """\
# MemoVex environment — edit and uncomment to enable optional features.
# Restart the stack after changes: docker compose up -d

# Enable dense vector embeddings (sentence-transformers, ~420 MB download on first run)
# EMBEDDINGS_ENABLED=true

# Enable embedded Chroma per-agent collection (no extra container)
# CHROMA_ENABLED=true

# Log verbosity: DEBUG | INFO | WARNING | ERROR
# LOG_LEVEL=INFO
"""


def copy_stack(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(DOCKER_DIR / "Dockerfile.standalone",        dest / "Dockerfile")
    shutil.copy2(DOCKER_DIR / "docker-compose.standalone.yml", dest / "docker-compose.yml")

    env_file = dest / ".env"
    if not env_file.exists():
        env_file.write_text(ENV_TEMPLATE)
        print(f"  Created  {env_file}")
    else:
        print(f"  Skipped  {env_file}  (already exists)")

    print(f"  Copied   {dest / 'Dockerfile'}")
    print(f"  Copied   {dest / 'docker-compose.yml'}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Install the MemoVex Docker stack into a permanent directory."
    )
    p.add_argument(
        "--dir",
        default=str(Path.home() / "memovex"),
        help="Destination directory (default: ~/memovex)",
    )
    p.add_argument(
        "--start",
        action="store_true",
        help="Run 'docker compose up -d --build' after copying",
    )
    args = p.parse_args()

    dest = Path(args.dir).expanduser().resolve()

    print(f"\nMemoVex Docker setup → {dest}\n")
    copy_stack(dest)

    print(f"""
Done. Your MemoVex stack is ready at:
  {dest}/

Start the stack:
  cd {dest}
  docker compose up -d

The cloned repository is no longer needed — you can safely delete it.
""")

    if args.start:
        print("Starting stack…")
        result = subprocess.run(
            ["docker", "compose", "up", "-d", "--build"],
            cwd=dest,
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
