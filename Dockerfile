# ── Stage 1: Builder ─────────────────────────────────────────────────────────

FROM python:3.10-slim AS builder



WORKDIR /app



# Install build dependencies (Replaced libgl1-mesa-glx with libgl1)

RUN apt-get update && apt-get install -y --no-install-recommends \

    gcc g++ libffi-dev libgl1 libglib2.0-0 \

    && rm -rf /var/lib/apt/lists/*



# Install Python deps into a virtual environment

COPY requirements.txt .

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \

    pip install --no-cache-dir -r requirements.txt



# ── Stage 2: Runtime ──────────────────────────────────────────────────────────

FROM python:3.10-slim AS runtime



WORKDIR /app



# Runtime system libs for OpenCV + PIL (Replaced libgl1-mesa-glx with libgl1)

RUN apt-get update && apt-get install -y --no-install-recommends \

    libgl1 libglib2.0-0 libgomp1 \

    && rm -rf /var/lib/apt/lists/*



# Copy venv from builder

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"



# Copy application code

COPY . .



# Create temp upload directory

RUN mkdir -p /tmp/smartzi_uploads



# Non-root user for security

RUN useradd -m -u 1000 smartzi && chown -R smartzi:smartzi /app /tmp/smartzi_uploads

USER smartzi



# Health check (matches standard port or defaults to 8080)

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \

    CMD python -c "import urllib.request, os; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8080') + '/health')"



# Port exposed dynamically by Cloud Run / Render

ENV PORT=8080

EXPOSE 8080



# Startup command utilizing Gunicorn with Uvicorn workers, bound to $PORT dynamically

CMD ["sh", "-c", "gunicorn -w 1 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT"] update it 

