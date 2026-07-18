FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py export_feedback.py ./
COPY static/ static/

# Tokenizer/model downloads and JSONL logs should not run as root's cache.
ENV HF_HOME=/tmp/hf
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
