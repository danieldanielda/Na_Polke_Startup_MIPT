FROM python:3.11-slim

WORKDIR /app

RUN apk update && apk gcc libc-dev g++ openssl libgcc tzdata icu-libc icu-data-full

COPY pyproject.toml .

RUN pip install uv
RUN uv pip install .

COPY main.py .
COPY crew.py .
COPY src ./src
COPY entrypoint.sh .

RUN sed -i 's/\r$//' ./entrypoint.sh && chmod +x ./entrypoint.sh

CMD ["/app/entrypoint.sh"]
