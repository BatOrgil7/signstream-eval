# Reproducible full-pipeline image. CPU-only execution (e.g. CI smoke) works
# on this base too; GPU execution requires the NVIDIA container runtime.
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Non-root user; the container never needs root after install.
RUN useradd --create-home --uid 1000 signstream

WORKDIR /workspace

COPY --chown=signstream:signstream . .

# torch comes from the base image (CUDA build); installing the pinned
# requirements.lock here would replace it with the PyPI wheel. The project
# and its remaining dependencies are installed on top of the base instead.
RUN pip install --no-cache-dir -e ".[dev]"

USER signstream

# Smoke experiment is the container's default check once the runner exists.
CMD ["python", "-m", "signstream.run", "experiment=smoke", "tracking=none"]
