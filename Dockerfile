FROM python:3.11-slim

WORKDIR /app

# Optimize building by loading framework dependencies early
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend python code along with index.html directly into working directory
COPY . .

# Hugging Face exposes traffic through port 7860
EXPOSE 7860

# Run Uvicorn ASGI mapping cleanly onto wildcard network host matching
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
