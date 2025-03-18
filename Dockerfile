FROM python:3.10.16-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    g++ 

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY pyproject.toml .
COPY uv.lock .

RUN uv sync
RUN uv pip install tensorflow==2.15.0

COPY . .

ENTRYPOINT ["uv", "run", "inference.py"]

CMD ["test_images/dog.png", "test_images/dog_rotated.png"]
