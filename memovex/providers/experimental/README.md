# `memovex.providers.experimental`

Adapters in this directory are **incomplete**. They expose part of an
external memory system to MemoVex, but at least one operation
(typically `store_memory`) is not wired end-to-end.

Do **not** register them with `MemoVexOrchestrator.register_provider()`
in production. They are kept in-tree so the design intent and partial
implementations remain visible while we finish them.

| Adapter             | Status                                                        |
| ------------------- | ------------------------------------------------------------- |
| `resonant_adapter`  | `store_memory` is a no-op pending a stable upstream API.      |

To use one anyway:

```python
from memovex.providers.experimental.resonant_adapter import ResonantAdapter
```
