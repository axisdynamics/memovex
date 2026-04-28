#!/usr/bin/env python3
"""
Generate .claude/settings.json with the correct absolute paths for a
memovex plugin.

Usage:
    python scripts/setup_plugin.py --agent claude
    python scripts/setup_plugin.py --agent claude --output /my/project/.claude/settings.json
    python scripts/setup_plugin.py --agent hermes
    python scripts/setup_plugin.py --agent openclaw
"""
import argparse
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent  # repo root


TEMPLATES = {
    "claude": {
        "hooks": {
            "UserPromptSubmit": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": None,  # filled below
                            "timeout": 8,
                            "statusMessage": "Consultando memoria [memovex]...",
                        }
                    ],
                }
            ],
            "Stop": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": None,  # filled below
                            "timeout": 15,
                            "statusMessage": "Guardando en memoria [memovex]...",
                        }
                    ],
                }
            ],
        }
    },
    "hermes": {
        "hooks": {
            "UserPromptSubmit": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": None,
                            "timeout": 8,
                            "statusMessage": "Consultando memoria [hermes]...",
                        }
                    ],
                }
            ],
            "Stop": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": None,
                            "timeout": 15,
                            "statusMessage": "Guardando en memoria [hermes]...",
                        }
                    ],
                }
            ],
        }
    },
    "openclaw": {
        "hooks": {
            "UserPromptSubmit": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": None,
                            "timeout": 8,
                            "statusMessage": "Consultando memoria [openclaw]...",
                        }
                    ],
                }
            ],
            "Stop": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": None,
                            "timeout": 15,
                            "statusMessage": "Guardando en memoria [openclaw]...",
                        }
                    ],
                }
            ],
        }
    },
}

HOOK_PATHS = {
    "claude": {
        "inject": HERE / "plugins" / "claude"   / "hooks" / "memory_inject.py",
        "store":  HERE / "plugins" / "claude"   / "hooks" / "memory_store.py",
    },
    "hermes": {
        "inject": HERE / "plugins" / "hermes"   / "hooks" / "memory_inject.py",
        "store":  HERE / "plugins" / "hermes"   / "hooks" / "memory_store.py",
    },
    "openclaw": {
        "inject": HERE / "plugins" / "openclaw" / "hooks" / "memory_inject.py",
        "store":  HERE / "plugins" / "openclaw" / "hooks" / "memory_store.py",
    },
}


def build_settings(agent: str) -> dict:
    paths = HOOK_PATHS[agent]
    cfg = json.loads(json.dumps(TEMPLATES[agent]))  # deep copy

    inject_cmd = f"python3 {paths['inject']}"
    store_cmd  = f"python3 {paths['store']}"

    cfg["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"] = inject_cmd
    cfg["hooks"]["Stop"][0]["hooks"][0]["command"] = store_cmd
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent",  required=True, choices=["claude", "hermes", "openclaw"])
    p.add_argument("--output", default=None,
                   help="Path to write settings.json (default: stdout)")
    args = p.parse_args()

    cfg = build_settings(args.agent)
    out = json.dumps(cfg, indent=2)

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out)
        print(f"Written → {path}")
    else:
        print(out)


if __name__ == "__main__":
    main()
