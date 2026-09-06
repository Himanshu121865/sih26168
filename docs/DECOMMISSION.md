# Decommission Checklist

Superseded paths are removed on a schedule, not left to rot. Each entry names
its replacement and the version that deletes it. Removing an entry = deleting
the code in the same PR (no dangling references).

| Path | Status | Replacement | Remove in |
|------|--------|-------------|-----------|
| `python/models/adapter.py` stub | Deprecated (F6) | `head_logsig_vel` σ head | v0.3, or when stage-2 trains a real adapter |
| `eval_drift.py --mode 1d` synthetic naive + `map*0.6` | screening-only | `--mode 2d` real ATE/RTE | keep until judges clear screening |
| `SeamlessHandler.rRampMs` (unused param) | Dead | — (delete param) | v0.2 |
| `MainActivity.downloadVisibleArea` (uncalled) | Dead | offline flow if revived | v0.2 |
| `releases/sih26168-debug.apk` in git | Migrating | GitHub Releases (CI) | first green Release |
| `print(` in CLI scripts | Removed (Tier 3) | `core/runlog.py` loguru | done — keep `download_iovnbd.py` on print (stdlib-only bootstrap) |
| `enum Mode` + `gnssTrust` field | Removed (ADR-007) | sealed `FusionMode` | done |

Rules: no new entry without a removal version; a removal PR deletes code +
this row together.
