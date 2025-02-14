FROM python:3.10-slim-bullseye AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY frontend/requirements.txt frontend/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r frontend/requirements.txt

FROM python:3.10-slim-bullseye

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN mkdir -p data/faiss_index

COPY config.py .
COPY .env .
COPY frontend/ frontend/
COPY data/faiss_index/*.faiss data/faiss_index/
COPY data/faiss_index/*.pkl data/faiss_index/

RUN useradd -m appuser && \
    chown -R appuser:appuser /app && \
    chmod 600 /app/.env && \
    chmod -R 755 data

ENV PYTHONPATH=/app

USER appuser

EXPOSE 8505

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8505", "--server.address=0.0.0.0"]
