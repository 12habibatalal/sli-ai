FROM docker.io/library/python:3.12-slim
WORKDIR /app
COPY --chmod=0644 requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chmod=0644 app.py .
USER 65534
ENV SLI_MODEL=/models/sentences-large.onnx SLI_VOCAB=/models/sentences-vocab.json SLI_PORT=8000
EXPOSE 8000
CMD ["python", "app.py"]
