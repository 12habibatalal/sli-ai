FROM docker.io/library/python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
USER 65534
ENV SLI_MODEL=/models/sentences-large.onnx SLI_VOCAB=/models/sentences-vocab.json SLI_PORT=8000
EXPOSE 8000
CMD ["python", "app.py"]
