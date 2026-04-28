"""
Hermes plugin — thin re-export of src.plugins.hermes_plugin.

Import from here for a stable, version-independent path:
    from plugins.hermes.plugin import create_hermes_memory, HermesMemoryPlugin
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from memovex.plugins.hermes_plugin import create_hermes_memory, HermesMemoryPlugin  # noqa: F401

__all__ = ["create_hermes_memory", "HermesMemoryPlugin"]
