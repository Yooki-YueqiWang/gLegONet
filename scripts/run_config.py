#!/usr/bin/env python3
"""Run a repository entry point from an explicit JSON configuration."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("Overrides must use KEY=VALUE")
    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Override keys cannot be empty")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value


def expand_value(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_value(item) for item in value]
    return value


def build_command(config: dict[str, Any], overrides: list[tuple[str, Any]]) -> list[str]:
    entrypoint = REPOSITORY_ROOT / config["entrypoint"]
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Configured entry point does not exist: {entrypoint}")
    arguments = dict(config.get("arguments", {}))
    for key, value in overrides:
        arguments[key] = value

    command = [sys.executable, str(entrypoint)]
    for key, raw_value in arguments.items():
        value = expand_value(raw_value)
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, list):
            value = ",".join(str(item) for item in value)
        command.extend([flag, str(value)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], type=parse_override)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError(f"Unsupported config schema in {config_path}")
    command = build_command(config, args.overrides)
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


if __name__ == "__main__":
    main()
