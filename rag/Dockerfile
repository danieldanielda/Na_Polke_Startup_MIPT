FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libc6-dev \
    g++ \
    libssl-dev \
    tzdata \
    libicu-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --no-cache-dir uv && \
    uv pip install --system .

COPY main.py .
COPY src ./src
COPY entrypoint.sh .

RUN sed -i 's/\r$//' ./entrypoint.sh && chmod +x ./entrypoint.sh

CMD ["/app/entrypoint.sh"]