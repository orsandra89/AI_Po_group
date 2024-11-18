FROM python:3.12-slim

WORKDIR /app

COPY ../requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY .. /app


ENV PYTHONPATH=/app

CMD ["python", "leaf_detection_ai/ai_detection.py"]