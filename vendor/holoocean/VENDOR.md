# Vendored HoloOcean client

| | |
|---|---|
| Upstream | `git@github.com:byu-holoocean/HoloOcean.git` |
| Tag | `v2.3.0` |
| Commit | `49e70552dfd97273b7dfbe755fbe65d7738b24b7` |
| License | MIT (see `LICENSE`) |

`client/src/holoocean/` is an **unmodified** copy of upstream at that commit. Verify with:

```
./scripts/update-vendor.sh
```

## What was kept

Only what `flake.nix` needs to build the `holoocean` Python package:

- `client/src/holoocean/` — the package (580 KB)
- `client/setup.py` — `setup()` reads `README.md` for `long_description`, so both are required
- `client/README.md`
- `client/pyproject.toml` — a 0-byte file, upstream's own; kept so this tree stays a
  faithful copy. `flake.nix` removes it at build time via `postPatch`, because an empty
  PEP 517 table confuses backend selection.
- `client/example.py` — upstream's sensor/agent usage reference
- `LICENSE`

## What was stripped

| Removed | Size | Why |
|---|---|---|
| `engine/`, `engine 5.6/` | 749 MB | Unreal C++ project. Needs a licensed Unreal Engine checkout to build; unusable here. Also under the **Epic EULA**, not MIT — see `LICENSE`. |
| `client/docs/` | 295 MB | Sphinx sources and images. Published at <https://byu-holoocean.github.io/holoocean-docs/>. |
| `client/tests/` | 7.9 MB | Upstream's simulator test suite; requires the 5.2 GB world binaries and a running Unreal process. |
| `docker/`, `.drone.yml`, `.github/`, `readthedocs.yml` | — | Upstream CI and packaging. |

## Why a copy and not a submodule

1. **Upstream is private.** `byu-holoocean/HoloOcean` is gated behind a GitHub account
   linked to an Epic Games account with the invitation accepted. A submodule would make
   this repository un-clonable for anyone — including CI — without that access.
2. **Cost.** A submodule costs ~601 MB (298 MB of `.git/modules`, plus docs and tests in
   the working tree) to deliver the 596 KB actually needed.
3. **Licensing.** The bulk of upstream is Unreal engine content under the Epic EULA. The
   part used here is MIT, which permits redistribution with attribution — that is what
   `LICENSE` and this file provide.
4. **Auditability is preserved.** `scripts/update-vendor.sh` reproduces the exact
   diff-against-tag check that a submodule pointer would have guaranteed.

## Updating

```
./scripts/update-vendor.sh v2.4.0     # check against a different tag
```

If it reports differences, re-copy `client/src`, `client/setup.py` and `client/README.md`
from that tag, then update the tag and commit in the table above.
