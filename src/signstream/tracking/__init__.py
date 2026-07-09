"""Experiment-tracker adapters behind a thin ``Tracker`` Protocol.

W&B and MLflow are interchangeable backends; the local filesystem remains
the source of truth (trackers mirror, never own, results). W&B runs in
offline mode by default and is synced manually; ``tracking=none`` is fully
supported.

Planned public interface: ``Tracker.log_metrics(d, step)``,
``Tracker.log_artifact(path)``, ``Tracker.finish()``, with ``WandbTracker``,
``MlflowTracker``, and ``NullTracker`` implementations.

Backend packages (wandb/mlflow) are optional dependencies; the adapters
import them lazily.
"""
