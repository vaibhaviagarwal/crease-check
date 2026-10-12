FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "uvicorn[standard]>=0.30"

COPY crease crease
COPY crease_api crease_api
COPY public public
COPY sql sql

EXPOSE 8000
CMD ["uvicorn", "crease_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
