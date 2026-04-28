"""
Claude Code plugin — thin re-export of src.plugins.claude_plugin.

Import from here for a stable, version-independent path:
    from plugins.claude.plugin import create_claude_memory, save_claude_memory
"""

import sys
import os

# Allow running from the plugins/claude/ directory directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from memovex.plugins.claude_plugin import create_claude_memory, save_claude_memory  # noqa: F401

__all__ = ["create_claude_memory", "save_claude_memory"]
