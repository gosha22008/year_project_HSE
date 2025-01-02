FROM python:3.10-slim

# Установка зависимостей системы
RUN apt-get update && apt-get install -y \
    build-essential \
    tk \
    libtk8.6 \
    libx11-dev \
    libxt-dev \
    libcurl4-openssl-dev \
    libcairo2-dev \
    libpango1.0-dev \
    gcc \
    pkg-config \ 
    libcairo2-dev \ 
    libpango1.0-dev \
    libglib2.0-dev \
    libmagic1 \
    python3-dev \ 
    libsystemd-dev \
    libdbus-1-dev \
    libgirepository1.0-dev \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
 
    # Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt
 
COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
