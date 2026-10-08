FROM python:3.11-slim

# libgomp1: LightGBM's native library links against the GNU OpenMP runtime,
# which python:*-slim does not ship. Without it `import lightgbm` raises
# "libgomp.so.1: cannot open shared object file" -- so every training and
# validation entrypoint in this image failed at import, not at run time.
# serving/Dockerfile already installed it; this one did not.
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=app:app . .

RUN mkdir -p data/raw data/processed data/reference reports/evidently models/champion models/challenger

USER app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
