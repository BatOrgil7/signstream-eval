"""Command-line entrypoint: ``python -m signstream.run stage=<name> experiment=<name>``.

Thin wrapper around :mod:`signstream.runner`. The stage graph and Hydra
wiring land in a later increment; until then this module only reserves the
entrypoint path used throughout the documentation.
"""


def main() -> None:
    """Dispatch to the runner stage graph."""
    raise NotImplementedError(
        "The experiment runner is not implemented yet; see signstream.runner "
        "for the planned stage graph."
    )


if __name__ == "__main__":
    main()
