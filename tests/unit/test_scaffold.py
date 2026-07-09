"""Scaffold sanity checks: packaging, package discovery, and frozen contract constants.

These tests guard the repository skeleton (src layout, package discovery,
dependency policy), not behavior; real unit tests arrive with the modules
they test.
"""

import importlib

import pytest

# Every package here must import without torch, mediapipe, or any other
# full-extra dependency — that is the dependency policy under test.
TORCH_FREE_IMPORTS = [
    "signstream",
    "signstream.schema",
    "signstream.data",
    "signstream.data.landmarks",
    "signstream.data.landmarks.schema",
    "signstream.alignment",
    "signstream.models",
    "signstream.streaming",
    "signstream.metrics",
    "signstream.stats",
    "signstream.viz",
    "signstream.runner",
    "signstream.tracking",
    "signstream.utils",
]


@pytest.mark.parametrize("name", TORCH_FREE_IMPORTS)
def test_package_imports_without_full_extra(name: str) -> None:
    importlib.import_module(name)


def test_version_declared() -> None:
    import signstream

    assert signstream.__version__


def test_schema_version_frozen() -> None:
    from signstream.schema import SCHEMA_VERSION

    assert SCHEMA_VERSION == "1.0"


def test_landmark_contract() -> None:
    from signstream.data.landmarks import schema as landmark_schema

    assert landmark_schema.N_TOTAL_LANDMARKS == 543
    assert landmark_schema.N_COORDS == 3
