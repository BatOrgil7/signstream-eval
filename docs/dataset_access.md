# Dataset access

No dataset — videos, landmarks, or annotations — is redistributed in this
repository. This page documents how to request or download each corpus and
where to place it (`$SIGNSTREAM_DATA`, see `configs/config.yaml`).

> Skeleton; completed as each dataset adapter lands.

## PHOENIX-2014T (primary, E1)

RWTH-PHOENIX-Weather 2014T. Ships as per-utterance folders of individual PNG
frames (`features/fullFrame-210x260px/{split}/{utt}/`) at a fixed 25 fps.
*Download/request instructions to be completed.*

## FSboard (optional, E2)

Fingerspelling subset, character-level transcripts. *To be completed.*

## ASL-MTP (stretch, E3)

Behind an access request; carries the non-manual tiers needed by the stretch
metrics. *To be completed.*

## Tinyset (CI fixture)

Synthetic; committed to git under `tests/fixtures/tinyset/`. No access
needed — regenerate with `scripts/make_tinyset.py`.
