FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# The project's only runtime dependencies are numpy and pyyaml.
RUN pip install --no-cache-dir numpy pyyaml

# Run from /app (not pip-installed): the server finds the web UI at ./web/dist
COPY cli.py ./
COPY src ./src
COPY web/dist ./web/dist
COPY serve.yaml ./serve.yaml
COPY start.sh ./start.sh
COPY scripts/fetch_model.py ./scripts/fetch_model.py
RUN chmod +x start.sh

# Model weights are NOT baked into the image. start.sh downloads them into
# MODEL_DIR (a persistent disk mounted at /data) on first boot.
ENV MODEL_DIR=/data/model

RUN useradd --create-home --uid 10001 ptf \
 && mkdir -p /data && chown -R ptf:ptf /app /data
USER ptf

EXPOSE 8000
CMD ["./start.sh"]
