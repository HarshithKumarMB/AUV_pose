# Vendored HoloOcean client

| | |
|---|---|
| Upstream | `git@github.com:byu-holoocean/HoloOcean.git` |
| Tag | `v2.3.0` |
| Commit | `49e70552dfd97273b7dfbe755fbe65d7738b24b7` |
| License | MIT (see `LICENSE`) |

An unmodified copy of upstream's `client/` at that commit, reduced to what
`flake.nix` builds: `src/holoocean/`, `setup.py`, `README.md` (read by
`setup.py`), the empty `pyproject.toml` (removed at build time) and
`example.py`.

It is a copy rather than a submodule because upstream is private (gated behind
an Epic Games account) and is mostly Unreal content under the Epic EULA; the
client is MIT. To update, copy the same files from the new tag and update the
table.
