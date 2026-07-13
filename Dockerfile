# Reproducible full-pipeline image. CPU-only execution (e.g. CI smoke) works
# on this base too; GPU execution requires the NVIDIA container runtime.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Non-root user; the container never needs root after install.
RUN useradd --create-home --uid 1000 signstream

WORKDIR /workspace

COPY --chown=signstream:signstream . .

# requirements.lock is applied as a constraints file so every dependency
# installs at its pinned version. The base image's torch 2.4.1 (CUDA build)
# already satisfies both the [full] range (>=2.4,<2.5) and the lock pin
# (torch==2.4.1), so it is left in place rather than re-downloaded.
RUN pip install --no-cache-dir -c requirements.lock -e ".[full,dev]"

USER signstream

# Smoke experiment is the container's default check once the runner exists.
CMD ["python", "-m", "signstream.run", "experiment=smoke", "tracking=none"]
