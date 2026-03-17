FROM python:3.12-slim

WORKDIR /app

# Установка системных зависимостей (комментарии только перед командой)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc-dev \
    openssl \
    libssl-dev \
    tzdata \
    libgl1 \
    libglib2.0-0 \
    libzbar0 \
    libzbar-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# Установка uv и зависимостей Python
RUN pip install --no-cache-dir uv && \
    uv pip install --system .

COPY bot.py .
COPY config.py .
COPY src ./src
COPY entrypoint.sh .

RUN chmod +x entrypoint.sh && \
    sed -i 's/\r$//' entrypoint.sh

CMD ["./entrypoint.sh"]