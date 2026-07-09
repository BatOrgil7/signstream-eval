#!/usr/bin/env python3
"""Regenerate the synthetic CI fixture in ``tests/fixtures/tinyset/``.

The tinyset is five synthetic utterances (seeded random-walk landmarks,
fabricated 3-6-gloss references, one crafted to exercise severity-class-A
tokens) committed to git so the integration smoke test runs on any machine
without licensed data. Total size must stay under 1 MB.

Not implemented yet: depends on the ``Sample`` types in
:mod:`signstream.data` and the emission-log schema.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "make_tinyset.py is implemented together with signstream.data; "
        "the committed fixture lives in tests/fixtures/tinyset/."
    )


if __name__ == "__main__":
    main()
