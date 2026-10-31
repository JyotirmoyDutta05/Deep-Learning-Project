FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY phishguard ./phishguard
COPY artifacts ./artifacts
ENV PHISHGUARD_MODEL=/app/artifacts/lstm/model.pt
RUN useradd -m app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u;u.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "phishguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
