FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 MODEL_DIR=/app/models HF_HOME=/tmp/hf ORT_THREADS=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY download_models.py .
RUN python download_models.py && rm -rf /tmp/hf
COPY kb.json .
COPY app ./app
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers ${WORKERS:-1}
