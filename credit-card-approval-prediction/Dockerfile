
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY pipeline.bin .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn predict:app --host 0.0.0.0 --port ${PORT:-10000}"]
