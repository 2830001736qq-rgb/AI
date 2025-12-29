FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

WORKDIR /app

# Install git for the diffusers dev branch
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]
