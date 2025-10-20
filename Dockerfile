# Video Composer Dockerfile - 4K Support
FROM python:3.11-slim

# Installeer FFmpeg
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Verificeer FFmpeg
RUN ffmpeg -version

# Werkdirectory
WORKDIR /app

# Installeer Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopieer applicatie
COPY main.py .

# Environment variabelen
ENV PORT=10000
ENV UVICORN_WORKERS=1
ENV PYTHONUNBUFFERED=1

# Expose poort
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT}/health').raise_for_status()" || exit 1

# Start applicatie
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}
