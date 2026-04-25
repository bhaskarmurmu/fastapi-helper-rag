FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet \
    && uv pip install --system -r requirements.txt

COPY src/ src/
COPY pyproject.toml .
RUN uv pip install --system -e .

EXPOSE 8080

CMD ["uvicorn", "fastapi_helper.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
