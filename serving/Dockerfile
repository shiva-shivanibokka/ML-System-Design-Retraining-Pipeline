# serving/Dockerfile — image for the FastAPI serving+dashboard API (HF Spaces)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app \
    HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

# LightGBM's native lib needs the GNU OpenMP runtime, absent from python:*-slim.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home app
COPY --chown=app:app . .
USER app

# HF Spaces routes to 7860; fall back to $PORT if set.
ENV PORT=7860
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT}"]
