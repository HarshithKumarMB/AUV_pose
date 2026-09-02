"""Runnable experiments built from :mod:`auv_pose`.

Scripts here own their HoloOcean scenarios, file paths and plotting; the algorithms
they compose live in ``auv_pose``. Run them from the repository root so relative
data paths resolve::

    python experiments/train_map.py

Anything that opens a world needs the simulator environment::

    nix run .#sim -- -c "python experiments/navigate.py"

These are not tests. Several call ``holoocean.make()`` at module scope, so pytest is
restricted to ``tests/`` (see ``pyproject.toml``) to keep collection from launching
the simulator.
"""
